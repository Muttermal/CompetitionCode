# !/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BigBirdModel

from utils import pooling


class BaseModel(nn.Module):

    def __init__(self, model_path: str, out_size: int, dropout_rate: float):
        super().__init__()
        self.pre_trained_model = BigBirdModel.from_pretrained(model_path)
        self.linear = nn.Linear(in_features=768, out_features=out_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        model_output = self.pre_trained_model(input_ids, attention_mask, token_type_ids)
        model_logits = pooling(model_output, attention_mask=attention_mask, type_="mean")
        out_result = self.linear(self.dropout(model_logits))
        return out_result


if __name__ == '__main__':
    my_model = BaseModel(model_path="../../huggingface/bigbird-roberta-base", out_size=6, dropout_rate=0.3)
    print(my_model.__repr__(), "Done!")
