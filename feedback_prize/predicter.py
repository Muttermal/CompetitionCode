# !/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import os
import pandas as pd
from transformers import BigBirdTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typing

from dataset import MyDataset, device, label_dict
from model import BaseModel
from trainer import Config
from utils import convert_batch

label_dict_env = {inx: label for label, inx in label_dict.items()}


def prediction(model_list: typing.Union[torch.nn.Module, list] = None,
               dataloader: DataLoader = None, out_path: str = None):
    predict_result = {"text_id": []}
    if isinstance(model_list, torch.nn.Module):
        model_list = [model_list]
    with torch.no_grad():
        for model in model_list:
            model.eval()
        with tqdm(enumerate(dataloader), total=len(dataloader), ascii=False, desc="prediction") as pbar:
            for step, batch in pbar:
                batch = convert_batch(batch)
                text_id = batch["text_id"]
                for model in model_list:
                    temp_result = np.zeros(shape=(Config.batch_size, Config.out_size))
                    curr_model_res = model(input_ids=batch["input_ids"],
                                           attention_mask=batch["attention_mask"],
                                           token_type_ids=batch["token_type_ids"])
                    temp_result += (5 * torch.sigmoid(curr_model_res).cpu().numpy())
                model_score = temp_result / len(model_list)
                for i, example_score in enumerate(model_score):
                    predict_result["text_id"].append(text_id[i])
                    for inx, score in enumerate(example_score):
                        predict_result.setdefault(label_dict_env[inx], []).append(score)
    predict_df = pd.DataFrame(predict_result)
    predict_df.to_csv(out_path, index=False)
    tqdm.write("Done!")


if __name__ == '__main__':
    tokenizer = BigBirdTokenizer.from_pretrained(Config.pre_train_model_path)
    trained_model = BaseModel(model_path=Config.pre_train_model_path,
                              out_size=Config.out_size,
                              dropout_rate=Config.dropout_rate)
    trained_model_list = []
    for model_name in os.listdir(Config.model_save_path):
        print(f"loading model: {model_name}")
        model_path = os.path.join(Config.model_save_path, model_name)
        trained_model.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))
        trained_model.to(device)
        trained_model_list.append(trained_model)

    eval_dataset = partial(MyDataset, seq_len=Config.seq_len, batch_size=Config.batch_size, tokenizer=tokenizer,
                           drop_last=False, shuffle=False)
    test_dataset = eval_dataset(data_path=Config.test_data_path, do_predict=True)
    test_dataloader = DataLoader(dataset=test_dataset)
    prediction(model_list=trained_model_list, dataloader=test_dataloader, out_path="./data/sample_test.csv")
