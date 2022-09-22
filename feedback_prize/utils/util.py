# !/usr/bin/env python
# -*- coding: utf-8 -*-

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import pandas as pd
import torch
import typing


def convert_batch(batch: typing.Union[list, dict]) -> dict:
    if isinstance(batch, dict):
        result = batch
    else:
        result = {"text_id": [i["text_id"][0] for i in batch]}  # tqdm会增加一个[]，导致tensor shape多了一维
        used_keys = {k for k in batch[0].keys() if k != "text_id"}
        for key in used_keys:
            result[key] = torch.stack([i[key][0] for i in batch], dim=0).view(len(batch), -1)
    return result


def data_split(file_path: str, split_size: typing.Union[int, float] = 0.8,
               random_seed: int = 2022, mode: str = "", target_cols: list = None):

    raw_data = pd.read_csv(file_path)
    if mode == "k_fold":    # 将数据集交叉划分
        if not isinstance(split_size, int) or split_size <= 1:
            raise ValueError(f"split_size should be integer and greater than 1, but got {split_size}")
        if not target_cols:
            target_cols = raw_data.columns
        data_folder = MultilabelStratifiedKFold(n_splits=split_size, shuffle=True, random_state=random_seed)
        for fold, (_, val_) in enumerate(data_folder.split(X=raw_data, y=raw_data[target_cols])):
            raw_data.loc[val_, "k_fold"] = int(fold)
        raw_data["k_fold"] = raw_data["k_fold"].astype(int)
        return raw_data
    else:   # 简单划分为训练集和验证集
        if split_size > 1:
            raise ValueError(f"split_size should be less than 1, but got {split_size}")
        train_data = raw_data.sample(frac=split_size, random_state=random_seed).reset_index(drop=True)
        val_data = raw_data.drop(train_data.index).reset_index(drop=True)
        return train_data, val_data


def load_model(model: torch.nn.Module, model_saved_dir: str):
    if os.path.isdir(model_saved_dir):
        model_saved_dir = os.path.join(model_saved_dir, "model.bin")
    model.load_state_dict(torch.load(model_saved_dir, map_location="cpu"))
    return model
