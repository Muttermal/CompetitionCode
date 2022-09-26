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
               random_seed: int = 2022, target_cols: list = None):

    raw_data = pd.read_csv(file_path)
    if 0 < split_size < 1:  # 正常划分
        train_data = raw_data.sample(frac=split_size, random_state=random_seed).reset_index(drop=True)
        val_data = raw_data.drop(train_data.index).reset_index(drop=True)
        return train_data, val_data
    elif split_size == 1:   # 全部作为训练集或验证集
        return raw_data.copy()
    else:   # 交叉验证K折划分
        split_size = int(split_size)
        target_cols = raw_data.columns if not target_cols else target_cols
        data_folder = MultilabelStratifiedKFold(n_splits=split_size, shuffle=True, random_state=random_seed)
        for fold, (_, val_) in enumerate(data_folder.split(X=raw_data, y=raw_data[target_cols])):
            raw_data.loc[val_, "k_fold"] = int(fold)
        raw_data["k_fold"] = raw_data["k_fold"].astype(int)
        return raw_data


def load_model(model: torch.nn.Module, model_saved_dir: str):
    if os.path.isdir(model_saved_dir):
        model_saved_dir = os.path.join(model_saved_dir, "model.bin")
    model.load_state_dict(torch.load(model_saved_dir, map_location="cpu"))
    return model
