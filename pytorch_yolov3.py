# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/23 22:06
@Auth ： killbulala
@File ：pytorch_yolov3.py
@IDE ：PyCharm
@Email：killbulala@163.com
darknet53 yolov3 inp_size: 416
"""
import torch
import torch.nn as nn


# 卷积模块
class ConvolutionalLayer(nn.Module):

    def __init__(self, inp, oup, k, s, p):
        super(ConvolutionalLayer, self).__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.sub(x)
        return output


# 下采样的卷积
class DownsampleLayer(nn.Module):

    def __init__(self, inp):
        super(DownsampleLayer, self).__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(inp, inp*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inp*2),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.sub(x)
        return output


# 上采样模块
class UpsampleLayer(nn.Module):

    def __init__(self, ):
        super(UpsampleLayer, self).__init__()
        self.sub = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        output = self.sub(x)
        return output


# 残差模块
class ResidualLayer(nn.Module):

    def __init__(self, inp):
        super(ResidualLayer, self).__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(inp, inp // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inp // 2),
            nn.ReLU(),
            nn.Conv2d(inp // 2, inp, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inp),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.sub(x)
        return x + output


# 卷积集合模块
class ConvolutionalSet(nn.Module):

    def __init__(self, inp):
        super(ConvolutionalSet, self).__init__()
        self.sub = nn.Sequential(
            ConvolutionalLayer(inp, inp // 2, 1, 1, 0),
            ConvolutionalLayer(inp // 2, inp, 3, 1, 1),
            ConvolutionalLayer(inp, inp // 2, 1, 1, 0),
            ConvolutionalLayer(inp // 2, inp, 3, 1, 1),
            ConvolutionalLayer(inp, inp // 2, 1, 1, 0),
        )

    def forward(self, x):
        output = self.sub(x)
        return output


class YoloV3(nn.Module):

    def __init__(self, num_classes=20):
        super(YoloV3, self).__init__()
        self.output_channel = 3 * (5 + num_classes)
        self.out_52x52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownsampleLayer(32),
            ResidualLayer(64),
            DownsampleLayer(64),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsampleLayer(128),
        )

        self.out_26x26 = nn.Sequential(
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            DownsampleLayer(256),
        )

        self.out_13x13 = nn.Sequential(
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            DownsampleLayer(512),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
        )

        self.set1 = ConvolutionalSet(1024)
        self.pred_13x13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, self.output_channel, 1, 1, 0)
        )

        self.set1_down = nn.Sequential(
            ConvolutionalLayer(512, 512, 3, 1, 1),  # 为了和26x26相拼接 改成512通道
            UpsampleLayer(),
        )
        self.set2 = ConvolutionalSet(1024)
        self.pred_26x26 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, self.output_channel, 1, 1, 0)
        )

        self.set2_down = nn.Sequential(
            ConvolutionalLayer(512, 256, 3, 1, 1),
            UpsampleLayer(),
        )
        self.set3 = ConvolutionalSet(512)
        self.pred_52x52 = nn.Sequential(
            ConvolutionalLayer(256, 1024, 3, 1, 1),
            nn.Conv2d(1024, self.output_channel, 1, 1, 0)
        )

    def forward(self, x):
        output_52x52 = self.out_52x52(x)
        output_26x26 = self.out_26x26(output_52x52)
        output_13x13 = self.out_13x13(output_26x26)

        set1_out = self.set1(output_13x13)
        pred_13x13 = self.pred_13x13(set1_out)

        set1_down = self.set1_down(set1_out)
        concatenate1 = torch.cat((output_26x26, set1_down), dim=1)
        set2_out = self.set2(concatenate1)
        pred_26x26 = self.pred_26x26(set2_out)

        set2_down = self.set2_down(set2_out)
        concatenate2 = torch.cat((output_52x52, set2_down), dim=1)
        set3_out = self.set3(concatenate2)
        pred_52x52 = self.pred_52x52(set3_out)
        return pred_52x52, pred_26x26, pred_13x13


if __name__ == '__main__':
    yolov3 = YoloV3()
    x = torch.randn(1, 3, 416, 416)
    output1, output2, output3 = yolov3(x)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)

