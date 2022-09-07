# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import pandas as pd
import time
import torch
import typing


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        """Reset timer."""
        self.running = True
        self.total = 0
        self.start = time.time()

    def resume(self):
        """Resume."""
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """Stop."""
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    @property
    def time(self):
        """Return time."""
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class AvgMeter:
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.avg, self.sum, self.count = [0] * 3

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class EarlyStopping:
    """Early Stopping"""

    def __init__(self, patience: int = None, key=None, reverse: bool = False):
        """Early stopping Constructor."""
        self._patience = patience
        self.key = key
        self.reverse = reverse
        self._best_so_far = 1000 if self.reverse else 0
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = False
        self._early_stop = False

    def state_dict(self):
        """A `Trainer` can use this to serialize the state."""
        return {
            'patience': self._patience,
            'best_so_far': self._best_so_far,
            'is_best_so_far': self._is_best_so_far,
            'epochs_with_no_improvement': self._epochs_with_no_improvement,
        }

    def load_state_dict(self, state_dict) -> None:
        """Hydrate a early stopping from a serialized state."""
        self._patience = state_dict["patience"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._best_so_far = state_dict["best_so_far"]
        self._epochs_with_no_improvement = state_dict["epochs_with_no_improvement"]

    def update(self, score: typing.Union[float, dict]):
        """Call function."""
        if isinstance(score, dict):
            score = score[self.key]
        if self.reverse:
            if score <= self._best_so_far:
                self.better_than_before(score)
            else:
                self.bad_than_before()
        else:
            if score > self._best_so_far:
                self.better_than_before(score)
            else:
                self.bad_than_before()

    def better_than_before(self, score: float):
        self._best_so_far = score
        self._is_best_so_far = True
        self._epochs_with_no_improvement = 0

    def bad_than_before(self):
        self._is_best_so_far = False
        self._epochs_with_no_improvement += 1

    @property
    def best_so_far(self) -> int:
        """Returns best so far."""
        return self._best_so_far

    @property
    def is_best_so_far(self) -> bool:
        """Returns true if it is the best so far."""
        return self._is_best_so_far

    @property
    def should_stop_early(self) -> bool:
        """Returns true if improvement has stopped for long enough."""
        if not self._patience:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience


def pooling(hidden_tensor, attention_mask: torch.Tensor, type_: str = "mean", has_cls: bool = True):
    """获取embedding vector经过不同pooling方式后的结果"""

    assert type_ in {"mean", "max", "cls", "mean_sqrt_len_tokens"}
    if not has_cls and type_ == "cls":
        raise ValueError("hidden tensor does not has `cls`.")

    token_embeddings, cls_token_embeddings = (hidden_tensor[0],  hidden_tensor[1]) if has_cls else (hidden_tensor, None)

    if type_ == "cls":
        out_tensor = cls_token_embeddings
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


def convert_batch(batch: typing.Union[list, dict]) -> dict:
    if isinstance(batch, dict):
        result = batch
    else:
        result = {"text_id": [i["text_id"][0] for i in batch]}  # tqdm会增加一个[]，导致tensor shape多了一维
        used_keys = {k for k in batch[0].keys() if k != "text_id"}
        for key in used_keys:
            result[key] = torch.stack([i[key][0] for i in batch], dim=0).view(len(batch), -1)
    return result


def file_split(file_path: str, split_size: float = 0.8, out_file: list = None):
    raw_data = pd.read_csv(file_path)
    train_data = raw_data.sample(frac=split_size, random_state=200)  # random state is a seed value
    test_data = raw_data.drop(train_data.index)
    # split_mask = np.random.rand(len(raw_data)) < split_size
    # train_data, test_data = raw_data[split_mask], raw_data[~split_mask]
    train_data.to_csv(out_file[0])
    test_data.to_csv(out_file[1])


def set_logger(log_file: str = 'log.txt', level: str = "info"):
    assert level.upper() in {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"}
    log_level = getattr(logging, level.upper())
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(filename=log_file, mode="a", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
