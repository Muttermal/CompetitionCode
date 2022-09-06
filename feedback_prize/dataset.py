# !/usr/bin/env python
# -*- coding: utf-8 -*-
import typing

import pandas as pd
import random
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers.tokenization_utils import PreTrainedTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = {"cohesion": 0, "syntax": 1, "vocabulary": 2, "phraseology": 3, "grammar": 4, "conventions": 5}


class MyDataset(IterableDataset):
    def __init__(self, data: typing.Union[str, pd.DataFrame], tokenizer: PreTrainedTokenizer, seq_len: int,
                 batch_size: int, drop_last: bool = False, shuffle: bool = True, do_predict: bool = False, **kwargs):
        super(MyDataset, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data_size, self.used_data = self.get_data()
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.do_predict = do_predict
        self.__dict__.update(**kwargs)

    def get_data(self, data_keys: list = None):
        if isinstance(self.data, pd.DataFrame):
            data = self.data
        elif isinstance(self.data, str):
            data = pd.read_csv(self.data)
        else:
            raise ValueError(f"The type of data should be str or pd.DataFrame, but got {type(self.data)}")
        data_size = data.shape[0]
        data_keys = data_keys if data_keys else data.columns.tolist()
        used_data = data.loc[:, data_keys].to_dict()
        return data_size, used_data

    def __iter__(self):
        batch = []
        all_data_inx = list(range(self.data_size))
        if self.shuffle:
            random.shuffle(all_data_inx)
        for inx in all_data_inx:
            temp_data = {k: self.used_data[k][inx] for k in self.used_data}
            tokenize_result = self.tokenizer(temp_data["full_text"], max_length=self.seq_len,
                                             padding='max_length', truncation=True)
            input_ids = torch.tensor(tokenize_result["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(tokenize_result["attention_mask"], dtype=torch.float, device=device)
            token_type_ids = torch.zeros_like(input_ids, device=device)
            if not self.do_predict:
                labels = torch.tensor([temp_data["cohesion"], temp_data["syntax"], temp_data["vocabulary"],
                                       temp_data["phraseology"], temp_data["grammar"], temp_data["conventions"]],
                                      device=device)
                example = dict(
                    text_id=temp_data["text_id"], label=labels, input_ids=input_ids,
                    attention_mask=attention_mask, token_type_ids=token_type_ids
                )
            else:
                example = dict(text_id=temp_data["text_id"], input_ids=input_ids,
                               attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch.append(example)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        dataset_length = self.data_size // self.batch_size
        if not self.drop_last:
            dataset_length += 1
        return dataset_length
