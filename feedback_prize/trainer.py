# !/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import partial
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import prettytable as pt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup, AdamW, BigBirdTokenizer

from model import BaseModel
from dataset import MyDataset, device
from utils import Timer, AvgMeter, EarlyStopping, convert_batch
from eval import validation, mcr_mse


class Config:
    # model
    pre_train_model_path: str = "../../pre_trained_model/NLP/bigbird-roberta-base"
    out_size: int = 6
    dropout_rate: float = 0.3
    # data
    batch_size: int = 2
    seq_len: int = 3000
    num_workers: int = 0
    train_data_path: str = "./data/ori_train.csv"
    val_data_path: str = "./data/validation.csv"
    test_data_path: str = "./data/test.csv"
    target_cols: str = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    n_fold: int = 5
    # training
    weight_decay: float = 1e-6
    lr: float = 1e-5
    epochs: int = 8
    metric_key: str = "MCRMSE"
    patience: int = 5
    # save
    model_save_path: str = "./saved_model/version_0906/"
    # logging:
    log_file: str = "train_log.txt"


if __name__ == '__main__':
    log_file = open(Config.log_file, "a")
    train_data = pd.read_csv(Config.train_data_path)
    data_folder = MultilabelStratifiedKFold(n_splits=Config.n_fold, shuffle=True)
    for fold, (_, val_) in enumerate(data_folder.split(X=train_data, y=train_data[Config.target_cols])):
        train_data.loc[val_, "k_fold"] = int(fold)

    train_data["k_fold"] = train_data["k_fold"].astype(int)

    tokenizer = BigBirdTokenizer.from_pretrained(Config.pre_train_model_path)
    my_model = BaseModel(model_path=Config.pre_train_model_path,
                         out_size=Config.out_size,
                         dropout_rate=Config.dropout_rate)
    my_model.to(device)

    dataset = partial(MyDataset, seq_len=Config.seq_len, batch_size=Config.batch_size, tokenizer=tokenizer)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in my_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': Config.weight_decay},
        {'params': [p for n, p in my_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr, eps=1e-8)

    criteria = torch.nn.MSELoss()

    for fold in range(Config.n_fold):
        early_stopping = EarlyStopping(patience=Config.patience, key=Config.metric_key, reverse=True)
        timer = Timer()
        tqdm.write(f"========== Fold: {fold} ==========")
        df_train = train_data[train_data.k_fold != fold].reset_index(drop=True)
        df_valid = train_data[train_data.k_fold == fold].reset_index(drop=True)
        train_dataset = dataset(data=df_train)
        val_dataset = dataset(data=df_valid)
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=Config.num_workers)
        val_dataloader = DataLoader(dataset=val_dataset, num_workers=Config.num_workers)
        total_steps = len(train_dataloader) * Config.epochs
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                              num_training_steps=total_steps)

        for epoch in range(Config.epochs):
            train_loss = AvgMeter("Train Loss")
            my_model.train()
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=False) as pbar:
                for step, input_data in pbar:
                    my_model.train()
                    input_data = convert_batch(input_data)
                    labels = input_data["label"]
                    model_res = my_model(input_ids=input_data["input_ids"],
                                         attention_mask=input_data["attention_mask"],
                                         token_type_ids=input_data["token_type_ids"])
                    model_score = 5 * torch.sigmoid(model_res)
                    loss = criteria(model_res.view(-1, Config.out_size), labels).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.update(loss.item())

                    pbar.set_description(f"Epoch {epoch + 1}/{Config.epochs}")
                    pbar.set_postfix(loss=f'{loss.item():.3f}')

            scheduler.step()
            tqdm.write(f'Cost time: {timer.time}s')

            eval_res = validation(model=my_model, dataloader=val_dataloader, criteria=criteria, metric=mcr_mse)
            table = pt.PrettyTable(['fold', 'epoch', "MCRMSE", "Train_Loss"])
            table.add_row([fold + 1, epoch + 1, eval_res["MCRMSE"], train_loss.avg])
            tqdm.write(s=table.get_string(), file=log_file)
            early_stopping.update(score=eval_res)
            if early_stopping.is_best_so_far:
                torch.save(my_model.state_dict(), Config.model_save_path + f"model_{fold + 1}.bin")
                tqdm.write(f"current_metric: {Config.metric_key}, current best score: {early_stopping.best_so_far}, "
                           f"model saved {Config.model_save_path}.")
            if early_stopping.should_stop_early:
                tqdm.write("run out of patience, stop of training...")
                break
    log_file.close()
