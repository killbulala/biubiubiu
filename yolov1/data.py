# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/22 19:39
@Auth ： killbulala
@File ：data.py
@IDE ：PyCharm
@Email：killbulala@163.com
"""
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cfg import *


class Data_(Dataset):

    def __init__(self, root):
        super(Data_, self).__init__()
        with open(root, 'r') as f:
            self.lines = f.readlines()
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img, ann = self.get_annotations(idx)
        # data encoder
        img, target = self.encoder(img, ann)
        return img, target

    def __len__(self):
        return len(self.lines)

    def get_annotations(self, idx):
        tmp = self.lines[idx].strip().split(' ')
        img, scale = self.get_img(tmp[0], TRAIN_SIZE)
        ann = self.get_ann(tmp[1:], scale)
        # if self.transforms:
        #     ann = self.transforms(ann)
        return img, ann

    def get_img(self, path, train_size):
        img = Image.open(path)
        max_hw = max(img.size[0], img.size[1])
        img_mask = Image.new(mode='RGB', size=(max_hw, max_hw), color=(0, 0, 0))
        img_mask.paste(img, box=(0, 0))
        img_mask = img_mask.resize((train_size, train_size))
        scale = train_size / max_hw
        return img_mask, scale

    def get_ann(self, ann_lst, scale):
        box = np.array([np.array(list(map(int, box.split(','))), dtype=np.float64) for box in ann_lst])
        box[:, 0:4] *= scale
        box[:, 0:4] /= TRAIN_SIZE
        return box

    def encoder(self, img, ann):
        num_objs = ann.shape[0]
        target = torch.zeros((GRID_NUM, GRID_NUM, NUM_BBOX*5+CLASSES))  # 编码是 H W 30
        cell_size = 1. / GRID_NUM
        wh = ann[:, 2:4] - ann[:, :2]
        cxcy = (ann[:, :2] + ann[:, 2:4]) / 2
        for i in range(num_objs):
            obj_cls = int(ann[i][4])
            cell_xy_idx = np.array([math.ceil(x) - 1 for x in (cxcy[i][:2] / cell_size)])
            target[cell_xy_idx[1], cell_xy_idx[0], 4] = 1
            target[cell_xy_idx[1], cell_xy_idx[0], 9] = 1
            target[cell_xy_idx[1], cell_xy_idx[0], 10+obj_cls] = 1
            left_top_xy = cell_xy_idx * cell_size
            delta_xy = cxcy[i][:2] - left_top_xy
            cell_delta_xy = delta_xy / cell_size  # 在格子中的偏移
            target[cell_xy_idx[1], cell_xy_idx[0], :2] = torch.tensor(cell_delta_xy, dtype=torch.float32)
            target[cell_xy_idx[1], cell_xy_idx[0], 2:4] = torch.tensor(wh[i], dtype=torch.float32)
            target[cell_xy_idx[1], cell_xy_idx[0], 5:7] = torch.tensor(cell_delta_xy, dtype=torch.float32)
            target[cell_xy_idx[1], cell_xy_idx[0], 7:9] = torch.tensor(wh[i], dtype=torch.float32)

        img = self.transforms(img)
        return img, target


if __name__ == '__main__':
    r = r'F:\killbulala\work\datasets\VOCdevkit\2012_train.txt'
    data = Data_(r)
    print(data[0][0].shape)
    print(data[0][1].shape)
