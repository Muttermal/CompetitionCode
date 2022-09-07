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
from typing import Union

from model import BaseModel
from dataset import MyDataset, device
from utils import Timer, AvgMeter, EarlyStopping, convert_batch, set_logger
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


logger = set_logger(log_file=Config.model_save_path + "train_log.txt", level="info")


def main(config=Config, train_data: Union[str, pd.DataFrame] = None, val_data: Union[str, pd.DataFrame] = None):

    # 0.初始化工具
    early_stopping = EarlyStopping(patience=config.patience, key=config.metric_key, reverse=True)
    timer = Timer()

    # 1.准备模型和数据
    my_model = BaseModel(model_path=config.pre_train_model_path,
                         out_size=config.out_size,
                         dropout_rate=config.dropout_rate)
    my_model.to(device)
    tokenizer = BigBirdTokenizer.from_pretrained(config.pre_train_model_path)
    dataset = partial(MyDataset, seq_len=config.seq_len, batch_size=config.batch_size, tokenizer=tokenizer)
    train_dataset = dataset(data=train_data)
    val_dataset = dataset(data=val_data)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=config.num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, num_workers=config.num_workers)
    logger.info("model loaded.")

    # 2.optimizer, scheduler, criteria
    no_decay = ['bias', 'LayerNorm.weight']
    total_steps = len(train_dataloader) * config.epochs
    optimizer_grouped_parameters = [
        {'params': [p for n, p in my_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in my_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-8)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                          num_training_steps=total_steps)
    criteria = torch.nn.MSELoss()

    # 3.train and eval
    for epoch in range(config.epochs):
        train_loss = AvgMeter("Train Loss")
        my_model.train()
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=False) as pbar:
            for step, input_data in pbar:
                input_data = convert_batch(input_data)
                labels = input_data["label"]
                model_res = my_model(input_ids=input_data["input_ids"],
                                     attention_mask=input_data["attention_mask"],
                                     token_type_ids=input_data["token_type_ids"])
                model_score = 5 * torch.sigmoid(model_res)
                loss = criteria(model_score, labels).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.update(loss.item())

                pbar.set_description(f"Epoch {epoch + 1}/{config.epochs}")
                pbar.set_postfix(loss=f'{train_loss.avg:.3f}')

        scheduler.step()
        logger.info(f'Cost time: {timer.time}s')

        eval_res = validation(model=my_model, dataloader=val_dataloader, criteria=criteria, metric=mcr_mse)
        table = pt.PrettyTable(['fold', 'epoch', "MCRMSE", "Train_Loss"])
        table.add_row([fold + 1, epoch + 1, eval_res["MCRMSE"], train_loss.avg])
        logger.info(table.get_string())
        early_stopping.update(score=eval_res)
        if early_stopping.is_best_so_far:
            torch.save(my_model.state_dict(), config.model_save_path + f"model_{fold + 1}.bin")
            logger.info(f"current_metric: {config.metric_key}, current best score: {early_stopping.best_so_far}, "
                        f"model saved {config.model_save_path}.")
        if early_stopping.should_stop_early:
            logger.info("run out of patience, stop of training...")
            break


if __name__ == '__main__':
    raw_train_data = pd.read_csv(Config.train_data_path)
    data_folder = MultilabelStratifiedKFold(n_splits=Config.n_fold, shuffle=True)
    for fold, (_, val_) in enumerate(data_folder.split(X=raw_train_data, y=raw_train_data[Config.target_cols])):
        raw_train_data.loc[val_, "k_fold"] = int(fold)

    raw_train_data["k_fold"] = raw_train_data["k_fold"].astype(int)

    for fold in range(Config.n_fold):
        logger.info("="*15 + f"{fold}" + "="*15)
        df_train = raw_train_data[raw_train_data.k_fold != fold].reset_index(drop=True)
        df_valid = raw_train_data[raw_train_data.k_fold == fold].reset_index(drop=True)
        main(config=Config, train_data=df_train, val_data=df_valid)
