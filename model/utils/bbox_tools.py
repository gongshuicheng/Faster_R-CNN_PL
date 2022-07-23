# For Faster R-CNN
# import six  # six.moves.range for both python2(xrange) and python3(range)
# from six import __init__

import math
import torch
import torch.nn as nn

from data.data_utils import resize_img, normalize
from model.utils.no_grad import without_grad


@without_grad
def loc2bbox(src_bboxes, locs):
    if src_bboxes.shape[0] == 0:  # if there is no bbox
        return torch.zeros((0, 4), dtype=locs.dtype)
    
    device = locs.device
    
    src_height = src_bboxes[:, 2] - src_bboxes[:, 0]
    src_width = src_bboxes[:, 3] - src_bboxes[:, 1]
    src_ctr_y = src_bboxes[:, 0] + 0.5 * src_height
    src_ctr_x = src_bboxes[:, 1] + 0.5 * src_width

    dy = locs[:, 0::4]
    dx = locs[:, 1::4]
    dh = locs[:, 2::4]
    dw = locs[:, 3::4]
    
    # print(dy.shape)
    # print(src_height.shape)
    # print(src_ctr_y.shape)
    ctr_y = dy * src_height.unsqueeze(-1).to(device) + src_ctr_y.unsqueeze(-1).to(device)
    ctr_x = dx * src_width.unsqueeze(-1).to(device) + src_ctr_x.unsqueeze(-1).to(device)
    h = torch.exp(dh) * src_height.unsqueeze(-1).to(device)
    w = torch.exp(dw) * src_width.unsqueeze(-1).to(device)

    dst_bbox = torch.zeros(locs.shape, dtype=locs.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    
    dst_bbox = dst_bbox.to(device) 

    return dst_bbox


@without_grad
def bbox2loc(src_bbox, dst_bbox):
    
    device = src_bbox.device
    dst_bbox = dst_bbox.to(device)
    
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = torch.tensor([torch.finfo(height.dtype).eps]).to(device)
    height = torch.maximum(height, eps)
    width = torch.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = torch.log(base_height / height)
    dw = torch.log(base_width / width)

    locs = torch.vstack((dy, dx, dh, dw)).permute((1, 0))
    return locs.to(device)

@without_grad
def bbox_iou(bbox_a, bbox_b):
    """
    Calculate the Intersection of union
    Args:
        bbox_a:
        bbox_b:

    Returns:

    """
    
    # Device
    device = bbox_b.device
    bbox_a = bbox_a.to(device)
    
    # if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:  # Within one batch
    #     raise IndexError
    
    # bbox_b_one = bbox_b[0] if bbox_a.shape[1] != 4 else bbox_b
    # print(bbox_b_one.shape)


    # IoUs' top left
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # IoUs' bottom right
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(axis=2)  # Area of IoU
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)  # Area of a
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)  # Area of b
    return area_i / (area_a[:, None] + area_b - area_i)

@without_grad
def generate_anchor_base(
        base_size=16,  # Feature map unit size
        ratios=None,
        anchor_scales=None
):
    if anchor_scales is None:
        anchor_scales = [8, 16, 32]
    if ratios is None:
        ratios = [0.5, 1, 2]

    py = base_size / 2.
    px = base_size / 2.

    anchor_base = torch.zeros(
        (len(ratios) * len(anchor_scales), 4),
        dtype=torch.float32
    )
    
    # Area: 128^2, 256^2, 512^2
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * math.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * math.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    return anchor_base


if __name__ == "__main__":
    anchor_base = generate_anchor_base()
    print(anchor_base)
    print(f"Total {len(anchor_base)} anchor bases!")
    print(f"Total {len(anchor_base)} anchor bases!")