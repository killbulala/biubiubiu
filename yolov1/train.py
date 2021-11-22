# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/22 20:18
@Auth ： killbulala
@File ：train.py
@IDE ：PyCharm
@Email：killbulala@163.com
"""
import torch
import warnings
warnings.filterwarnings("ignore")
from net import YOLOv1_resnet
from data import Data_
from torch.utils.data import DataLoader
import torch.optim as optim
from cfg import LR, EPOCH, DEVICE
from loss import Yolov1Loss


net = YOLOv1_resnet().to(DEVICE)
dataset = Data_(r'F:\killbulala\work\datasets\VOCdevkit\2012_train.txt')
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
optim = optim.SGD(net.parameters(), lr=LR)
# optim = optim.SGD(net.parameters(), lr=LR)
criterion = Yolov1Loss()

total_loss = []
for epoch in range(EPOCH):
    for i, batch in enumerate(loader):
        img = batch[0].to(DEVICE)
        target = batch[1].to(DEVICE)
        output = net(img)
        loss = criterion.get_loss(output, target)
        total_loss.append(loss.item())
        if i % 50 == 0:
            print(f'Epoch: {epoch}  ==> iter: {i+1}  ==> loss: {loss}')
        optim.zero_grad()
        loss.backward()
        optim.step()

    torch.save(net.state_dict(), f'./pkls/yolov1_epoch_{epoch}.pth')

