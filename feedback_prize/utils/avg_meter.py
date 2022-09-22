# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 16:26
# @Author  : Zhang Guangyi
# @File    : avg_meter.py


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
