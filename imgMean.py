# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/16 20:49
@Auth ： killbulala
@File ：imgMean.py
@IDE ：PyCharm
@Email：killbulala@163.com

path: 训练数据的根目录
return : 各个通道的均值
均值是为了对图像进行标准化，可以移除图像的平均亮度值 (intensity)。
很多情况下我们对图像的照度并不感兴趣，而更多地关注其内容，
比如在对象识别任务中，图像的整体明亮程度并不会影响图像中存在的是什么物体。
这时对每个数据点移除像素的均值是有意义的[1]。
而另一个资料显示在每个样本上减去数据的统计平均值可以移除共同的部分，凸显个体差异
"""
import os
import cv2
import numpy as np

path = r'C:\Users\86131\Desktop\imgs'


def compute(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return int(R_mean), int(G_mean), int(B_mean)


if __name__ == '__main__':
    R, G, B = compute(path)
    print(R, G, B)


