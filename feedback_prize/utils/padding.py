# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 16:41
# @Author  : Zhang Guangyi
# @File    : padding.py
import numpy as np
import torch.nn.functional as F


def feat_padding(input_ids, attention_mask, token_label, batch_length, padding_dict, padding_side):
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()

    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids = input_ids.index_select(0, mask_index)
    token_label = token_label.index_select(0, mask_index)
    attention_mask = attention_mask.index_select(0, mask_index)
    ids_length = len(input_ids)

    if ids_length > batch_length:
        if random_seed <= 0.33:
            input_ids = input_ids[:batch_length]
            attention_mask = attention_mask[:batch_length]
            token_label = token_label[:batch_length]
        elif random_seed >= 0.66:
            input_ids = input_ids[-batch_length:]
            attention_mask = attention_mask[-batch_length:]
            token_label = token_label[-batch_length:]
        else:
            sub_length = ids_length - batch_length
            start_idx = np.random.randint(sub_length + 1)
            input_ids = input_ids[start_idx:start_idx + batch_length]
            attention_mask = attention_mask[start_idx:start_idx + batch_length]
            token_label = token_label[start_idx:start_idx + batch_length]

    if ids_length < batch_length:
        add_length = batch_length - ids_length
        if random_seed <= 0.33:
            input_ids = F.pad(input_ids, (0, add_length), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (0, add_length), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (0, add_length), "constant", padding_dict['input_ids'])
        elif random_seed >= 0.66:
            input_ids = F.pad(input_ids, (add_length, 0), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length, 0), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length, 0), "constant", padding_dict['input_ids'])
        else:
            add_length1 = np.random.randint(add_length + 1)
            add_length2 = add_length - add_length1
            input_ids = F.pad(input_ids, (add_length1, add_length2), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length1, add_length2), "constant",
                                   padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length1, add_length2), "constant", padding_dict['input_ids'])

    return input_ids, attention_mask, token_label


if __name__ == '__main__':
    input_ids = [1, 11672, 86, 806, 34, 57, 11511, 6, 8, 24, 18, 95, 562, 357, 2],
    token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    feat_padding(input_ids=input_ids, attention_mask=attention_mask, token_label="",
                 batch_length=20, padding_dict={"PADDING": 0}, padding_side="right")
