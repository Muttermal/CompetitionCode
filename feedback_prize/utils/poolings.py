# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 16:30
import torch


def pooling(hidden_tensor, attention_mask: torch.Tensor, type_: str = "mean"):
    """获取embedding vector经过不同pooling方式后的结果"""

    assert type_ in {"mean", "max", "cls", "mean_sqrt_len_tokens"}

    token_embeddings = hidden_tensor[0]
    cls_token_embeddings = hidden_tensor[1] if len(hidden_tensor) > 1 else None

    if type_ == "cls":
        if cls_token_embeddings:
            out_tensor = cls_token_embeddings
        else:
            raise ValueError("hidden_tensor has no cls_token_embedding")
    elif type_ == "max":
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        out_tensor = torch.max(token_embeddings, 1)[0]
    elif type_ in {"mean", "mean_sqrt_len_tokens"}:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        if type_ == "mean":
            out_tensor = sum_embeddings / sum_mask
        else:
            out_tensor = sum_embeddings / torch.sqrt(sum_mask)
    else:
        raise NotImplementedError
    return out_tensor
