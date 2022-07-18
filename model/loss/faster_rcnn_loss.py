import torch
from collections import namedtuple

LossTuple = namedtuple(
    "LossTuple",
    ["rpn_loc_loss",
     "rpn_cls_loss",
     "roi_loc_loss",
     "roi_cls_loss"]
)


def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).type(torch.float32)
    loss = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return loss.sum()


def fast_rcnn_loc_loss(pred_locs, gt_locs, gt_labels, sigma):
    in_weight = torch.zeros(gt_locs.shape)

    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_labels
    in_weight[(gt_labels > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = smooth_l1_loss(pred_locs, gt_locs, in_weight, sigma)

    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_labels >= 0).sum().type(torch.float32))  # ignore gt_label==-1 for rpn_loss
    return loc_loss
