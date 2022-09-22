# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
import os
import prettytable as pt
# from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer
# from typing import Union
import warnings
import yaml

from model import BaseModel
from dataset import MyDataset, device
from utils import Timer, AvgMeter, EarlyStopping, convert_batch, set_logger, data_split
from optimize import AWP
from eval import validation, mcr_mse

warnings.filterwarnings("ignore")


def main(parsers: argparse.Namespace):

    # 0.初始化工具
    if not os.path.exists(parsers.model_save_path):
        os.makedirs(parsers.model_save_path, exist_ok=True)
    timer = Timer()
    logger = set_logger(log_file=os.path.join(parsers.model_save_path, "train_log.txt"), level="info", name="main")
    logger.info("training parameters:")
    logger.info(parsers.__dict__)

    # 1.准备模型和数据
    my_model = BaseModel(model_path=parsers.pre_train_model_path,
                         out_size=parsers.out_size,
                         dropout_rate=parsers.dropout_rate)
    my_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(parsers.pre_train_model_path)
    dataset = partial(MyDataset, seq_len=parsers.seq_len, batch_size=parsers.batch_size, tokenizer=tokenizer)
    logger.info("model loaded.")

    # 2.optimizer, scheduler, criteria, AT(attack train)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in my_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': parsers.weight_decay},
        {'params': [p for n, p in my_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=parsers.lr, eps=1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=2, T_mult=2, eta_min=parsers.min_lr)
    scaler = torch.cuda.amp.GradScaler()
    awp = AWP(model=my_model, optimizer=optimizer, adv_lr=parsers.adv_lr,
              adv_eps=parsers.adv_eps, start_epoch=1, scaler=scaler)  # 对抗训练

    # 3.train and eval
    def do_training(train_data: DataLoader, eval_data: DataLoader):
        early_stopping = EarlyStopping(patience=parsers.patience, key=parsers.metric_key, reverse=True)
        if parsers.continue_training:
            loaded_model_path = os.path.join(parsers.model_save_path, "model.bin")
            if os.path.exists(loaded_model_path):
                my_model.load_state_dict(torch.load(loaded_model_path, map_location=device))
                logger.info(f"load model from {loaded_model_path}")

        for epoch in range(parsers.epochs):
            train_loss = AvgMeter("Train Loss")
            # writer = SummaryWriter(logdir=parsers.model_save_path, filename_suffix=f"{epoch+1}")
            my_model.train()
            with tqdm(enumerate(train_data), total=len(train_data), ascii=False) as pbar:
                for step, input_data in pbar:
                    input_data = convert_batch(input_data)
                    labels = input_data["label"]
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():  # 混合精度训练， 加快训练速度
                        model_res, loss = my_model(input_ids=input_data["input_ids"],
                                                   attention_mask=input_data["attention_mask"],
                                                   token_type_ids=input_data["token_type_ids"],
                                                   label=labels, pooling_type="mean")
                    # loss.backward()
                    # optimizer.step()
                    scaler.scale(loss).backward()
                    if parsers.enabel_awp:
                        awp.attack_backward(input_ids=input_data["input_ids"],
                                            attention_mask=input_data["attention_mask"],
                                            token_type_ids=input_data["token_type_ids"],
                                            label=labels,
                                            epoch=epoch)
                    torch.nn.utils.clip_grad_norm_(parameters=my_model.parameters(), max_norm=parsers.max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss.update(loss.item())
                    # writer.add_scalar(tag="loss", scalar_value=train_loss.avg, global_step=step,
                    #                   display_name=f"epoch_{epoch+1}_loss")
                    pbar.set_description(f"Epoch {epoch + 1}/{parsers.epochs}")
                    pbar.set_postfix(loss=f'{train_loss.avg:.4f}')

            scheduler.step()
            logger.info(f'Cost time: {timer.time}s')
            torch.cuda.empty_cache()

            eval_res = validation(model=my_model, dataloader=eval_data, metric=mcr_mse)
            table = pt.PrettyTable(['epoch', parsers.metric_key, "Train_Loss"])
            table.add_row([epoch + 1, eval_res[parsers.metric_key], train_loss.avg])
            logger.info("eval result:" + "\n" + table.get_string())
            early_stopping.update(score=eval_res)
            if early_stopping.is_best_so_far:
                torch.save(my_model.state_dict(), os.path.join(parsers.model_save_path, "model.bin"))
                logger.info(f"current_metric: {parsers.metric_key}, current best score: {early_stopping.best_so_far}, "
                            f"model saved {parsers.model_save_path}.")
            if early_stopping.should_stop_early:
                logger.info("run out of patience, stop of training...")
                break
        return early_stopping.best_so_far

    if hasattr(parsers, "n_fold") and parsers.n_fold > 1:   # 交叉训练
        logger.info("begin cross training")
        raw_data = data_split(file_path=parsers.train_data_path, random_seed=2022, mode="k_fold",
                              split_size=parsers.n_fold, target_cols=parsers.target_cols)
        eval_scores = []
        for fold in range(parsers.n_fold):
            logger.info("="*30 + f"fold {fold + 1 / parsers.n_fold}" + "="*30)
            df_train = raw_data[raw_data.k_fold != fold].reset_index(drop=True)
            df_valid = raw_data[raw_data.k_fold == fold].reset_index(drop=True)
            train_dataset = dataset(data=df_train)
            val_dataset = dataset(data=df_valid)
            train_dataloader = DataLoader(dataset=train_dataset, num_workers=parsers.num_workers)
            val_dataloader = DataLoader(dataset=val_dataset, num_workers=parsers.num_workers)
            curr_eval_score = do_training(train_data=train_dataloader, eval_data=val_dataloader)
            eval_scores.append(curr_eval_score)
        logger.info(f"cv score: {sum(eval_scores) / len(eval_scores)}")
    else:
        df_train, df_valid = data_split(file_path=parsers.train_data_path, split_size=0.8, random_seed=2022)
        train_dataset = dataset(data=df_train)
        val_dataset = dataset(data=df_valid)
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=parsers.num_workers)
        val_dataloader = DataLoader(dataset=val_dataset, num_workers=parsers.num_workers)
        eval_score = do_training(train_data=train_dataloader, eval_data=val_dataloader)
        logger.info(f"final eval score: {eval_score}")

    logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", type=str, default="configs/deberta_base.yaml", help="config file")
    args = parser.parse_args()
    yaml_params = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    for k, v in yaml_params.items():
        setattr(args, k, v)

    main(parsers=args)
