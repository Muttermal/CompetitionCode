# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 17:38
# @Author  : Zhang Guangyi
# @File    : attack_train.py

import torch
import torch.nn as nn


class AWP:
    """
    AWP: Adversarial weight perturbation helps robust generalization, https://arxiv.org/pdf/2004.05884.pdf
    :param model:
    :param optimizer:
    :param adv_param:
    :param adv_lr:
    :param adv_eps:
    :param start_epoch:
    :param adv_step:
    :param scaler:
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, adv_param: str = "weight",
                 adv_lr: float = 1., adv_eps: float = 0.2, start_epoch: int = 0, adv_step: int = 1, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, input_ids, attention_mask, token_type_ids, label, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                tr_logits, adv_loss = self.model(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 label=label)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
