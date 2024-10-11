import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        # pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss

# class SoftIoULoss(nn.Module):
#     def __init__(self, **kwargs):
#         super(SoftIoULoss, self).__init__()
#
#     def forward(self, pred, target):
#         # Old One
#         pred = torch.sigmoid(pred)
#         smooth = 1
#
#         # print("pred.shape: ", pred.shape)
#         # print("target.shape: ", target.shape)
#
#         intersection = pred * target
#         loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
#
#         # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
#         #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
#         #         - intersection.sum(axis=(1, 2, 3)) + smooth)
#
#         loss = 1 - loss.mean()
#         # loss = (1 - loss).mean()
#
#         return loss