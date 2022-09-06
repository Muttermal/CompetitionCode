# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 17:50
# @Author  : Zhang Guangyi
# @File    : eval.py
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable

from utils import AvgMeter, convert_batch


def mcr_mse(y_true, y_pred):
    scores = []
    idxes = len(y_true)
    for i in range(idxes):
        temp_y_true = y_true[i]
        temp_y_pred = y_pred[i]
        score = mean_squared_error(temp_y_true, temp_y_pred, squared=False)  # RMSE
        scores.append(score)
    mcr_mse_score = np.mean(scores)
    return {"MCRMSE": mcr_mse_score}


def validation(model: torch.nn.Module = None, dataloader: DataLoader = None,
               metric: Callable = None, criteria=None) -> dict:
    eval_loss = AvgMeter("Eval Loss")
    true_label = []
    model_result = []
    with torch.no_grad():
        model.eval()
        with tqdm(enumerate(dataloader), total=len(dataloader), ascii=False, desc="Validation") as pbar:
            for step, batch in pbar:
                batch = convert_batch(batch)
                labels = batch["label"]
                curr_model_res = model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"],
                                       token_type_ids=batch["token_type_ids"])
                if criteria:
                    curr_loss = criteria(curr_model_res, labels)
                    eval_loss.update(curr_loss.item())
                true_label.extend(labels.cpu().numpy())
                model_result.extend(curr_model_res.cpu().numpy())
    eval_result = metric(y_pred=model_result, y_true=true_label)
    if eval_loss.avg > 0:
        eval_result.update({"loss": eval_loss.avg})
    return eval_result
