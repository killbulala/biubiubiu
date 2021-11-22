# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/22 19:42
@Auth ： killbulala
@File ：cfg.py
@IDE ：PyCharm
@Email：killbulala@163.com
"""
import torch


NUM_BBOX = 2
CLASSES = 20
TRAIN_SIZE = 448
GRID_NUM = 7
LR = 0.0001
EPOCH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COOBJ = 5.
NOOBJ = .5
