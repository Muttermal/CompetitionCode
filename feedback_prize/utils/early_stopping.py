# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 16:27
# @Author  : Zhang Guangyi
# @File    : early_stopping.py
import typing


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
