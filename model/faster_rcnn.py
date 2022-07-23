from __future__ import absolute_import
from __future__ import division

import random
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from torchvision.ops import nms

from data.data_utils import preprocess
from model.utils.bbox_tools import bbox_iou, loc2bbox, bbox2loc, generate_anchor_base
from model.utils.init_tools import init_layer
from model.utils.creators import AnchorTargetCreator, ProposalTargetCreator

from model.utils.no_grad import without_grad


class ProposalCreator(object):
    def __init__(
            self,
            parent_model,
            nms_thresh=0.7,
            n_train_pre_nms=12000,
            n_train_post_nms=2000,
            n_test_pre_nms=6000,
            n_test_post_nms=300,
            min_size=16
    ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    
    @without_grad
    def __call__(self, locs, scores, anchors, img_size, scale):
        
        device = locs.device
        
        # Transform all regression locations to bboxes
        rois = loc2bbox(anchors, locs)
        rois[:, 0:4:2] = torch.clamp(rois[:, 0:4:2], 0, img_size[0])
        rois[:, 1:4:2] = torch.clamp(rois[:, 1:4:2], 0, img_size[1])
        
        # Drop the small rois under the minimum size
        min_size = self.min_size * scale
        hs = rois[:, 2] - rois[:, 0]
        ws = rois[:, 3] - rois[:, 1]
        
        keep = torch.where((hs >= min_size) & (ws >= min_size))[0]
        rois = rois[keep, :]
        scores = scores[keep]

        # NOTE: when test, remember faster_rcnn.eval() to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Sort with scores and choose N
        order = scores.ravel().argsort(dim=-1, descending=True)
        if n_pre_nms > 0:
            order = order[: n_pre_nms]
        rois = rois[order, :]
        scores = scores[order]

        # NMS
        keep = nms(rois, scores, self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        rois = rois[keep]
        return rois


class RegionProposalNetwork(nn.Module):  # RPN
    def __init__(
            self,
            rpn_in_channels=512, rpn_h_channels=512,
            ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
            feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()

        self.anchor_base = generate_anchor_base(
            base_size=feat_stride, anchor_scales=anchor_scales, ratios=ratios
        )
        n_anchors = self.anchor_base.shape[0]
        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        # Conv. 3x3 512
        self.conv1 = nn.Conv2d(rpn_in_channels, rpn_h_channels, (3, 3), stride=(1, 1), padding=1)
        # Conv. 1x1 18
        self.score_layer = nn.Conv2d(rpn_h_channels, n_anchors * 2, (1, 1), stride=(1, 1), padding=0)
        # Conv. 1x1 36
        self.loc_layer = nn.Conv2d(rpn_h_channels, n_anchors * 4, (1, 1), stride=(1, 1), padding=0)

        init_layer(self.conv1, mean=0, std=0.01)
        init_layer(self.score_layer, mean=0, std=0.01)
        init_layer(self.loc_layer, mean=0, std=0.01)

    def forward(self, x, img_size, scale):  # x should be in the form of the batch
        n, _, fm_height, fm_width = x.shape
        
        # Hidden Layer: 3x3 Conv. 512 => output Nx512x(H/16)x(W/16)
        h = F.relu(self.conv1(x))
        
        # Locations: 1x1 Conv. 4 output => output Nx4x(H/16)x(W/16)
        rpn_locs = self.loc_layer(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        
        # Scores: 1x1 Conv. n_anchors * 2 = 18 => output Nx18x(H/16)x(W/16)
        rpn_scores = self.score_layer(h)

        shifted_anchors = self._enumerate_shifted_anchor(
            self.anchor_base,
            self.feat_stride,
            fm_height, fm_width
        )

        n_anchors = self.anchor_base.shape[0]

        rpn_softmax_scores = F.softmax(rpn_scores.view(n, fm_height, fm_width, n_anchors, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        # ROI
        rois = list()
        roi_idx = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i], rpn_fg_scores[i], shifted_anchors, img_size, scale
            )
            batch_idx = i * torch.ones((len(roi),), dtype=torch.int32)
            rois.append(roi)
            roi_idx.append(batch_idx)
        rois = torch.cat(rois, dim=0)
        roi_idx = torch.cat(roi_idx, dim=0)
        
        # print(rpn_locs.shape, rpn_scores.shape, rois.shape, roi_idx.shape, shifted_anchors.shape)
        # rpn_locs with the shape of (1, n_anchors, 4)
        # rpn_scores with the shape of (1, n_anchors, 2)
        # rois with the shape of (2000, 4)
        # roi_idx with the shape of (2000)
        # shifted_anchors with the shape of (n_anchors, 4)
         
        return rpn_locs[0], rpn_scores[0], rois, roi_idx, shifted_anchors

    @staticmethod
    def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # return (K*A, 4)

        # Initialize
        # from x to y
        shift_y = torch.arange(0, height * feat_stride, feat_stride)
        shift_x = torch.arange(0, width * feat_stride, feat_stride)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift = torch.stack(
            (shift_y.ravel(), shift_x.ravel(),
             shift_y.ravel(), shift_x.ravel()),
            axis=1
        )

        n_anchors = anchor_base.shape[0]
        n_cells = shift.shape[0]
        anchors = anchor_base.view((1, n_anchors, 4)) + shift.view((1, n_cells, 4)).permute((1, 0, 2))
        anchors = anchors.view((n_cells * n_anchors, 4))
        return anchors


########################################################################################################################
# The Faster R-CNN
class FasterRcnn(nn.Module):
    def __init__(
            self,
            extractor, rpn, head,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRcnn, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.n_classes = self.head.n_classes

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        # mean and std of the locs
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset("evaluate")
        
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def use_preset(self, preset):
        if preset == "visualize":
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == "evaluate":
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def forward(self, imgs, bboxes, labels, scale):
        
        # print("self.device :", self.device)
        # print("######### In Faster R-CNN ##############")
        # print("imgs: ", imgs.device)
        # print("bboxes: ", bboxes.device)
        # print("labels: ", labels.device)
        
        img_size = imgs.shape[2:]  # imgs (torch.tensor): [batch_size, C, H, W]
        
        # Extract Features
        features = self.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_idx, shifted_anchors = self.rpn(features, img_size, scale)

        # RPN Tragets
        gt_rpn_locs, gt_rpn_labels = self.anchor_target_creator(
            bboxes, shifted_anchors, img_size
        )
        
        # Proposal Targets
        roi_samples, gt_roi_locs, gt_roi_labels = self.proposal_target_creator(
            rois, bboxes, labels,
            self.loc_normalize_mean,
            self.loc_normalize_std
        )
        roi_sample_idx = torch.zeros(len(roi_samples))
        
        
        # roi_sample_idx = torch.zeros(len(roi_samples)).to(self.device)
        
        # print("features: ", features.device)
        # print("roi_samples:", roi_samples.device)
        # print("roi_sample_idx", roi_sample_idx.device)
        
        roi_cls_locs, roi_scores = self.head(features, roi_samples, roi_sample_idx)

        return (
            rpn_locs, rpn_scores,
            gt_rpn_locs, gt_rpn_labels,
            roi_cls_locs, roi_scores,
            gt_roi_locs, gt_roi_labels,
            roi_samples
        )
    
    def predict(self, imgs, scale):
        # self.eval()
        # print(f"Model RPN Training: {self.rpn.training}")  # False
        
        img_size = imgs.shape[2:]
        
        # Scale on cuda but torch.tensor(img_size) on cpu
        original_size = torch.tensor(img_size, dtype=torch.float32).to(scale.device) / scale

        
        # Extract Features
        # rois: torch.Size([n_post_nms, 4])
        # roi_cls_locs: torch.Size([n_post_nms, n_classes*4])
        features = self.extractor(imgs)
        _, _, rois, roi_idx, _ = self.rpn(features, img_size, scale)
        roi_cls_locs, roi_scores = self.head(features, rois, roi_idx)
        
        # Original Locs
        # mean/std: torch.Size([1, n_classes*4])
        # original_roi_cls_locs: torch.Size([n_post_nms, n_classes, 4])
        # rois: torch.Size([n_post_nms, n_classes, 4])
        # mean = torch.tensor(self.loc_normalize_mean).cuda()[None]
        # std = torch.tensor(self.loc_normalize_std).cuda()[None]
        mean = torch.tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_classes)[None]
        std = torch.tensor(self.loc_normalize_std).cuda(). \
            repeat(self.n_classes)[None]
        original_roi_cls_locs = (roi_cls_locs * std + mean)
        original_roi_cls_locs = original_roi_cls_locs.view(-1, self.n_classes, 4)  # for rois expanding
        rois = rois.view(-1, 1, 4).expand_as(original_roi_cls_locs).contiguous().view(-1, 4) 
        original_roi_cls_locs = original_roi_cls_locs.view(-1, 4)

        
        original_roi_cls_bboxes = loc2bbox(rois, original_roi_cls_locs)
        
        # Clip BBOX
        original_roi_cls_bboxes[:, 0::2] = (original_roi_cls_bboxes[:, 0::2]).clamp(min=0, max=original_size[0])
        original_roi_cls_bboxes[:, 1::2] = (original_roi_cls_bboxes[:, 1::2]).clamp(min=0, max=original_size[1])
        
        prob = F.softmax(roi_scores, dim=-1)
        
        pred_bboxes, pred_labels, pred_scores = self._suppress(original_roi_cls_bboxes, prob)
        
        return pred_bboxes, pred_labels, pred_scores
    
    
    def _suppress(self, raw_cls_bboxes, raw_probs):
        bboxes = []
        labels = []
        scores = []
        for i in range(1, self.n_classes):
            cls_bbox_i = raw_cls_bboxes.reshape((-1, self.n_classes, 4))[:, i, :]  # torch.Size([n_post_nms, 4])
            prob_i = raw_probs[:, i]
            mask = prob_i > self.score_thresh  # First use score threshold
            masked_cls_bbox_i = cls_bbox_i[mask]
            masked_prob_i = prob_i[mask]
            keep = nms(masked_cls_bbox_i, masked_prob_i, self.nms_thresh)  # Second use NMS threshold
    
            bboxes.append(masked_cls_bbox_i[keep])
            # The labels are in [0, self.n_class - 2].
            labels.append((i-1) * torch.ones((len(keep),)))
            scores.append(masked_prob_i[keep])
    
        bboxes = torch.cat(bboxes, dim=0).type(torch.float32).to(self.device)  # torch.Size([n_suppressed_bboxes, 4])
        labels = torch.cat(labels, dim=0).type(torch.int32).to(self.device)  # torch.Size([n_suppressed_bboxes])
        scores = torch.cat(scores, dim=0).type(torch.float32).to(self.device)  # torch.Size(n_suppressed_bboxes])
        return bboxes, labels, scores
