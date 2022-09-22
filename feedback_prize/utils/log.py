# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 16:30
# @Author  : Zhang Guangyi
# @File    : logging.py
import logging


def set_logger(log_file: str = 'log.txt', level: str = "info", name: str = None):
    assert level.upper() in {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"}
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=log_level,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(name)
    return logger
