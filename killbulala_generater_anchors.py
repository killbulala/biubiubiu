# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/17 21:29
@Auth ： killbulala
@File ：killbulala_generater_anchors.py
@IDE ：PyCharm
@Email：killbulala@163.com
实现faster-rcnn中anchors
128 = 16 x 8
256 = 16 x 16
512 = 16 x 32
"""
import numpy as np


def xyxy2xywh(anchor_xyxy):
    w = anchor_xyxy[2] - anchor_xyxy[0] + 1
    h = anchor_xyxy[3] - anchor_xyxy[1] + 1
    cx = anchor_xyxy[0] + 0.5 * (w - 1)
    cy = anchor_xyxy[1] + 0.5 * (h - 1)
    return cx, cy, w, h


def xywh2xyxy(cx, cy, w, h):
    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    xmin = cx - 0.5 * (w - 1)
    ymin = cy - 0.5 * (h - 1)
    xmax = cx + 0.5 * (w - 1)
    ymax = cy + 0.5 * (h - 1)
    anchors = np.hstack((xmin, ymin, xmax, ymax))
    return anchors


def enum_scales_ratios(ratios_anchor, scales):
    cx, cy, w, h = xyxy2xywh(ratios_anchor)
    new_w = w * scales
    new_h = h * scales
    anchors = xywh2xyxy(cx, cy, new_w, new_h)
    return anchors


def get_ratios_wh(w, h, ratios):
    s_base_anchor = w * h
    ratios_anchors_w = np.array([np.round(np.sqrt(s_base_anchor / r)) for r in ratios])
    ratios_anchors_h = np.round(ratios_anchors_w * ratios)
    return ratios_anchors_w, ratios_anchors_h


def generator_anchors(base_size=16, ratios=np.array([0.5, 1, 2]), scales=np.array([8, 16, 32])):
    # 首先确定基础的anchor xmin ymin xmax ymax [0 0 15 15]
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    # 根据三种ratio生成三种宽高比的anchor  注意与base_anchor的面积保持一致且对应cx cy w h 格式
    cx, cy, w, h = xyxy2xywh(base_anchor)
    # 获取不同ratios下的w和h
    ratios_anchors_w, ratios_anchors_h = get_ratios_wh(w, h, ratios)
    # 还原anchor变现形式 xmin ymin xmax ymax 不同ratios变换得到 3 种anchors
    ratios_anchors = xywh2xyxy(cx, cy, ratios_anchors_w, ratios_anchors_h)
    # 在经过 3 种不同scales变化得到 9 种anchors  注意 scales是相对于宽高的比率变化
    scales_ratios_anchors = np.vstack([enum_scales_ratios(ratios_anchors[i, :], scales) for i in range(len(ratios))])
    print(scales_ratios_anchors.shape)

generator_anchors()


