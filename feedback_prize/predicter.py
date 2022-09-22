# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typing
import yaml

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
                temp_result = []
                for model in model_list:
                    curr_model_res, _ = model(input_ids=batch["input_ids"],
                                              attention_mask=batch["attention_mask"],
                                              token_type_ids=batch["token_type_ids"])
                    temp_result.append(curr_model_res.cpu().numpy())
                model_score = np.mean(temp_result, axis=0)
                for i, example_score in enumerate(model_score):
                    predict_result["text_id"].append(text_id[i])
                    for inx, score in enumerate(example_score):
                        predict_result.setdefault(label_dict_env[inx], []).append(score)
    predict_df = pd.DataFrame(predict_result)
    predict_df.to_csv(out_path, index=False)
    tqdm.write("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", type=str, default="configs/deberta_base.yaml", help="config file")
    args = parser.parse_args()
    yaml_params = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    for k, v in yaml_params.items():
        setattr(args, k, v)

    tokenizer = AutoTokenizer.from_pretrained(args.pre_train_model_path)
    trained_model = BaseModel(model_path=args.pre_train_model_path,
                              out_size=args.out_size,
                              dropout_rate=args.dropout_rate)
    trained_model_list = []
    for model_name in os.listdir(args.model_save_path):
        if model_name.endswith(".bin"):
            print(f"loading model: {model_name}")
            model_path = os.path.join(args.model_save_path, model_name)
            trained_model.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))
            trained_model.to(device)
            trained_model_list.append(trained_model)

    test_dataset = partial(MyDataset, seq_len=args.seq_len, batch_size=args.batch_size, tokenizer=tokenizer,
                           drop_last=False, shuffle=False)
    test_dataset = test_dataset(data=args.test_data_path, do_predict=True)
    test_dataloader = DataLoader(dataset=test_dataset)
    prediction(model_list=trained_model_list, dataloader=test_dataloader, out_path="./data/sample_test.csv")
