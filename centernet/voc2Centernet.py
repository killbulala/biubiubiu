# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/21 13:01
@Auth ： killbulala
@File ：dataset2.py
@IDE ：PyCharm
@Email：killbulala@163.com
数据处理时, 确保有目标并且目标框的宽高 > 0
"""
import math
from PIL import Image
import numpy as np
from torchvision import transforms
from utils import *
from torch.utils.data import Dataset, DataLoader


class Cdata(Dataset):

    def __init__(self, root, inp_size=512, num_class=20, isTrain=True):
        super(Cdata, self).__init__()
        self.inp_size=int(inp_size)
        self.oup_size=int(self.inp_size/4)
        self.num_class = int(num_class)
        self.isTrain = isTrain
        with open(root, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        tmp = self.lines[idx].strip().split(' ')
        # 数据增强
        image, box = self.get_data(tmp, self.inp_size)

        # 数据encoder
        batch_hm = np.zeros((self.oup_size, self.oup_size, self.num_class), dtype=np.float32)
        batch_wh = np.zeros((self.oup_size, self.oup_size, 2), dtype=np.float32)
        batch_reg = np.zeros((self.oup_size, self.oup_size, 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.oup_size, self.oup_size), dtype=np.float32)

        boxes = np.array(box[:, :4], dtype=np.float32)
        # 将原图的数据映射到输出特征图上 并防止越界
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.inp_size * self.oup_size, 0, self.oup_size-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.inp_size * self.oup_size, 0, self.oup_size-1)

        for i in range(len(box)):
            # 取出映射到预测特征图尺寸的信息 和类别信息
            bbox = boxes[i, :].copy()
            cls_id = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            # 映射真是值在特征图点 中心点编码 w h
            ct = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            # 绘制热力图 [128, 128, 17]  目标属于17类别的[128, 128]热力图
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
            # batch_wh编码是 w h顺序
            batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
            # 中心点偏移量
            batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
            # 偏移mask置为1
            batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))
        image = toTensor(image)
        batch_hm = toTensor(batch_hm)
        batch_wh = toTensor(batch_wh)
        batch_reg = toTensor(batch_reg)
        batch_reg_mask = toTensor(batch_reg_mask)
        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask

    def __len__(self):
        return len(self.lines)

    def get_data(self, tmp, inp_size):
        image = Image.open(tmp[0])
        image = cvtColor(image)
        # 解码真实值
        box = np.array([np.array(list(map(int,box.split(',')))) for box in tmp[1:]])
        # 将输入图像resize到训练尺寸,多余部分添加黑条
        image, scale = resize_img2train(image, inp_size)
        return image, box


if __name__ == '__main__':
    r = r'F:\killbulala\work\datasets\VOCdevkit\2012_train.txt'
    data = Cdata(r, isTrain=False)
    loader = DataLoader(data, batch_size=64, shuffle=False, drop_last=True)
    for i, batch in enumerate(loader):
        image, batch_hm, batch_wh, batch_reg, batch_reg_mask = batch
        print(str(i) + '*'*400)
        print(image.shape)
        print(batch_hm.shape)
        print(batch_wh.shape)
        print(batch_reg.shape)
        print(batch_reg_mask.shape)
