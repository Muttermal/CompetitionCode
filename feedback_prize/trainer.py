# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
import os
import pandas as pd
import prettytable as pt
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer
from typing import Union
import warnings
import yaml

from model import BaseModel
from dataset import MyDataset, device
from utils import Timer, AvgMeter, EarlyStopping, convert_batch, set_logger, data_split
from optimize import AWP
from eval import validation, mcr_mse

warnings.filterwarnings("ignore")


class Trainer:

    def __init__(self, parsers: argparse.Namespace):
        self.parsers = parsers
        self.tool_prepared()
        self.model_reset()

    def model_reset(self):
        self.model = self.get_model()
        self.optimizer, self.scheduler, self.scaler, self.awp = self.get_optimizer(self.model)

    def tool_prepared(self) -> None:
        if not os.path.exists(self.parsers.model_save_path):
            os.makedirs(self.parsers.model_save_path, exist_ok=True)
        self.timer = Timer()
        self.writer = SummaryWriter(logdir=self.parsers.model_save_path)
        self.logger = set_logger(log_file=os.path.join(self.parsers.model_save_path, "train_log.txt"),
                                 level="info", name="Trainer")
        self.logger.info("training parameters:")
        self.logger.info(self.parsers.__dict__)
        self.early_stopping = EarlyStopping(patience=self.parsers.patience, key=self.parsers.metric_key, reverse=True)

    def get_model(self) -> torch.nn.Module:
        model = BaseModel(model_path=self.parsers.pre_train_model_path,
                          out_size=self.parsers.out_size,
                          dropout_rate=self.parsers.dropout_rate)
        if self.parsers.continue_training:
            if os.path.exists(self.parsers.model_load_path):
                model.load_state_dict(torch.load(self.parsers.model_load_path, map_location="cpu"))
                self.logger.info(f"load model from {self.parsers.model_load_path}")
        model.to(device)
        self.logger.info("model loaded.")
        return model

    def get_data(self, train_data: Union[pd.DataFrame, str], eval_data: Union[pd.DataFrame, str]) -> tuple:
        tokenizer = AutoTokenizer.from_pretrained(self.parsers.pre_train_model_path)
        dataset = partial(MyDataset, seq_len=self.parsers.seq_len,
                          batch_size=self.parsers.batch_size, tokenizer=tokenizer)
        train_dataset = dataset(data=train_data)
        val_dataset = dataset(data=eval_data)
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=self.parsers.num_workers)
        val_dataloader = DataLoader(dataset=val_dataset, num_workers=self.parsers.num_workers)
        return train_dataloader, val_dataloader

    def get_optimizer(self, model: torch.nn.Module):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.parsers.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.parsers.lr, eps=1e-8)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=2, T_mult=2, eta_min=self.parsers.min_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        awp = AWP(model=model, optimizer=optimizer, adv_lr=self.parsers.adv_lr,
                  adv_eps=self.parsers.adv_eps, start_epoch=0, scaler=scaler,
                  adv_step=2)  # 对抗训练
        return optimizer, scheduler, scaler, awp

    def main(self, train_data: DataLoader, eval_data: DataLoader, version: Union[int, str] = ""):
        self.early_stopping.reset()
        if not self.parsers.continue_training:
            self.model_reset()  # 防止交叉训练时，当前模型接着上一折继续训练

        total_steps = 0
        self.logger.info("="*30 + f"begin training, version: {version}" + "="*30)
        for epoch in range(self.parsers.epochs):
            train_loss = AvgMeter("Train Loss")
            self.model.train()
            with tqdm(enumerate(train_data), total=len(train_data), ascii=False) as pbar:
                for step, input_data in pbar:
                    input_data = convert_batch(input_data)
                    labels = input_data["label"]

                    with torch.cuda.amp.autocast():  # 混合精度训练， 加快训练速度
                        model_res, loss = self.model(input_ids=input_data["input_ids"],
                                                     attention_mask=input_data["attention_mask"],
                                                     token_type_ids=input_data["token_type_ids"],
                                                     label=labels, pooling_type="mean")
                        loss = loss / self.parsers.gradient_accumulation_steps   # 梯度累积

                    self.scaler.scale(loss).backward()
                    # self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.parsers.max_norm)

                    if self.parsers.enabel_awp:
                        self.awp.attack_backward(input_ids=input_data["input_ids"],
                                                 attention_mask=input_data["attention_mask"],
                                                 token_type_ids=input_data["token_type_ids"],
                                                 label=labels,
                                                 epoch=epoch)

                    if (step+1) % self.parsers.gradient_accumulation_steps:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                    train_loss.update(loss.item())
                    self.writer.add_scalar(tag="train/loss", scalar_value=train_loss.avg, global_step=total_steps)
                    pbar.set_description(f"Epoch {epoch + 1}/{self.parsers.epochs}")
                    pbar.set_postfix(loss=f'{train_loss.avg:.4f}')
                    total_steps += 1

            self.scheduler.step()
            self.writer.add_scalar(tag="lr", scalar_value=self.optimizer.param_groups[0]['lr'], global_step=epoch)
            self.logger.info(f'Cost time: {self.timer.time}s')
            torch.cuda.empty_cache()

            eval_res = validation(model=self.model, dataloader=eval_data, metric=mcr_mse)
            table = pt.PrettyTable(['epoch', self.parsers.metric_key, "Train_Loss"])
            table.add_row([epoch + 1, eval_res[self.parsers.metric_key], train_loss.avg])
            self.logger.info("eval result:" + "\n" + table.get_string())
            self.early_stopping.update(score=eval_res)
            if self.early_stopping.is_best_so_far:
                torch.save(self.model.state_dict(), os.path.join(self.parsers.model_save_path, f"model_{version}.bin"))
                self.logger.info(f"current_metric: {self.parsers.metric_key}, "
                                 f"current best score: {self.early_stopping.best_so_far}, "
                                 f"model saved {self.parsers.model_save_path}.")
            if self.early_stopping.should_stop_early:
                self.logger.info("run out of patience, stop of training...")
                break
        return self.early_stopping.best_so_far

    def train(self):
        if self.parsers.n_fold == 1 and not os.path.exists(self.parsers.eval_data_path):
            raise ValueError("evaluate data not exist")

        if self.parsers.n_fold > 1:  # 交叉训练
            self.logger.info("begin cross training")
            raw_data = data_split(file_path=self.parsers.train_data_path, random_seed=2022,
                                  split_size=self.parsers.n_fold, target_cols=self.parsers.target_cols)
            eval_scores = []
            for fold in range(self.parsers.n_fold):
                df_train = raw_data[raw_data.k_fold != fold].reset_index(drop=True)
                df_valid = raw_data[raw_data.k_fold == fold].reset_index(drop=True)
                train_dataloader, val_dataloader = self.get_data(df_train, df_valid)
                curr_eval_score = self.main(train_data=train_dataloader, eval_data=val_dataloader, version=fold)
                eval_scores.append(curr_eval_score)
            final_eval_score = sum(eval_scores) / len(eval_scores)
        else:
            if self.parsers.n_fold == 1:
                df_train = data_split(file_path=self.parsers.train_data_path)
                df_valid = data_split(file_path=self.parsers.eval_data_path)
            else:
                df_train, df_valid = data_split(file_path=self.parsers.train_data_path, random_seed=2022,
                                                split_size=self.parsers.n_fold)
            train_dataloader, val_dataloader = self.get_data(df_train, df_valid)
            final_eval_score = self.main(train_data=train_dataloader, eval_data=val_dataloader,
                                         version=self.parsers.version)

        self.logger.info(f"final eval score: {final_eval_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", type=str, default="configs/deberta_base.yaml", help="config file")
    args = parser.parse_args()
    yaml_params = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    for k, v in yaml_params.items():
        setattr(args, k, v)

    trainer = Trainer(parsers=args)
    trainer.train()
