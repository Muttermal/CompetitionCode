# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from utils import pooling


class BaseModel(nn.Module):

    def __init__(self, model_path: str, out_size: int, dropout_rate: list):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        self.pre_trained_model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in dropout_rate])
        self.output = nn.Linear(config.hidden_size, out_size)
        self.criteria = torch.nn.MSELoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor, label: torch.Tensor = None, pooling_type: str = "mean"):
        model_output = self.pre_trained_model(input_ids, attention_mask, token_type_ids)
        model_logits = pooling(model_output, attention_mask=attention_mask, type_=pooling_type)
        out_result = [self.output(dropout_head(model_logits)) for dropout_head in self.dropout]
        out_result = sum(out_result) / len(self.dropout)
        loss = None
        if label is not None:
            loss = self.criteria(out_result, label).mean()
        return out_result, loss


if __name__ == '__main__':
    my_model = BaseModel(model_path="../../pre_trained_model/NLP/bigbird-roberta-base",
                         out_size=6, dropout_rate=[0.1, 0.2, 0.3, 0.4, 0.5])
    print(my_model.__repr__())