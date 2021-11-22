# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/22 20:25
@Auth ： killbulala
@File ：loss.py
@IDE ：PyCharm
@Email：killbulala@163.com
yolov1 loss implement
首先yolov1的损失都是l2损失, 包含四个部分,源码五个部分
1).原来有   预测有 中心xy损失
2).原来有   预测有 宽高wh平方根损失
3).原来没有 预测有 置信度confidence损失
4).原来有         类别损失
5).两个box中, iou小的confidence损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from cfg import GRID_NUM, NUM_BBOX, COOBJ, NOOBJ


class Yolov1Loss(nn.Module):

    def __init__(self):
        self.S = GRID_NUM
        self.B = NUM_BBOX
        self.co_obj = COOBJ
        self.no_obj = NOOBJ

    def compute_iou(self, box1, box2):
        """
        compute the iou of two boxes
        box1: tensor(N, 4) x1 y1 x2 y2
        box2: tensor(M, 4) x1 y1 x2 y2
        return iou tensor (N, M)
        """
        N = box1.shape[0]
        M = box2.shape[0]

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt
        wh[wh < 0] = 0  # clip
        inter = wh[:, :, 0] * wh[:, : 1]  # [N, M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        area1 = area1.unsqueeze(1).expand_as(inter)  # ==> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # ==> [N, M]

        iou = inter / (area1 + area2)
        return iou

    def get_loss(self, pred_tensor, target_tensor):
        """
        pred_tensor: [N, S, S, B*5+CLASSES]  5-> x y w h c
        target_tensor: [N, S, S, B*5+CLASSES]
        """
        batch_size = pred_tensor.shape[0]
        coo_mask = target_tensor[:, :, :, 4] == 1
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        """不包含目标"""
        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # 包含目标的预测结果
        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        cls_pred = coo_pred[:, 10:]
        # 包含目标的标注结果
        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        coo_len = box_target.size()[0]
        cls_target = coo_target[:, 10:]
        # ===> 包含目标的xy  wh  cls损失
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()).bool()
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0, coo_len, 2):
            box1 = box_pred[i:i+2]  # 取预测的前两个
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2] / self.S - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / self.S + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / self.S - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / self.S + 0.5 * box2[:, 2:4]
            # box1 [2, 4]
            # box2 [1, 4]
            iou = self.compute_iou(box1[:, :4], box2[:, :4])   # [2, 1]
            max_iou, max_idx = iou.max(0)
            # 将与真实框iou最大的加入cpp_response_mask  另一个加入coo_not_response_mask
            coo_response_mask[i+max_idx] = 1
            coo_not_response_mask[i+1-max_idx] = 1
            box_target_iou[i+max_idx, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()

        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)
        # 2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 3.class loss
        class_loss = F.mse_loss(cls_pred, cls_target, size_average=False)

        loss = (self.co_obj*loc_loss + 2*contain_loss + not_contain_loss + self.no_obj*nooobj_loss + class_loss) / batch_size
        return loss



















