import torch
import torch.nn as nn
import torch.functional as F

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    mask = torch.ge(gt_cls, 0)
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)

    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls, 0.6).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))

class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.landmark_factor = landmark_factor

        self.cls_loss = nn.BCELoss()
        self.box_loss = nn.MSELoss()
        self.landmark_loss = nn.MSELoss()

    def cls_loss_fn(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)

        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_prob_label = torch.masked_select(pred_label, mask)
        return self.cls_loss(valid_prob_label, valid_gt_label) * self.cls_factor

    def box_loss_fn(self, gt_label, gt_offsets, pred_offsets):
        gt_label = torch.squeeze(gt_label)
        gt_offsets = torch.squeeze(gt_offsets)
        pred_offsets = torch.squeeze(pred_offsets)

        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_offsets = gt_offsets[chose_index, :]
        valid_pred_offsets = pred_offsets[chose_index, :]

        return self.box_loss(valid_pred_offsets, valid_gt_offsets) * self.box_factor

    def landmark_loss_fn(self, gt_label, gt_landmark, pred_landmark):
        gt_label = torch.squeeze(gt_label)
        gt_landmark = torch.squeeze(gt_landmark)
        pred_landmark = torch.squeeze(pred_landmark)

        mask = torch.eq(gt_label, -2)
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.landmark_loss(valid_pred_landmark, valid_gt_landmark) * self.landmark_factor

