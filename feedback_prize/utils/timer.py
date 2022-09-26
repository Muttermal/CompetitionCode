# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 16:29
# @Author  : Zhang Guangyi
# @File    : timer.py
import time


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.reset()

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
