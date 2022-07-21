import torch
import torch.nn as nn
from torchnet.meter import ConfusionMeter, AverageValueMeter

from model.utils.creators import AnchorTargetCreator, ProposalTargetCreator
from model.loss.faster_rcnn_loss import LossTuple
from model.faster_rcnn_vgg16 import FasterRcnnVgg16


class FasterRcnnTrainer(nn.Module):
    def __int__(self, faster_rcnn=FasterRcnnVgg16(), rpn_sigma=3., roi_sigma=1., **kwargs):
        """
        Initialize the faster rcnn model and the creators.
        Args:
            faster_rcnn:
            rpn_sigma:
            roi_sigma:

        Returns:

        """
        # super(FasterRcnnTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter for k in LossTuple.fields}

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError("Currently only batch size 1 is supported!")

        H, W = imgs.shape[2], imgs.shape[3]
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_idx, shifted_anchors = \
            self.faster_rcnn.rpn(features, img_size, scale)

        bboxes = bboxes.squeeze(dim=0)
        labels = labels.squeeze(dim=0)
        rpn_scores = rpn_scores.squeeze(dim=0)
        rpn_locs = rpn_locs.squeeze(dim=0)

        # RPN Ground Truth
        gt_rpn_locs, gt_rpn_labels = self.anchor_target_creator(
            bboxes, shifted_anchors, img_size
        )

        roi_samples, gt_roi_locs, gt_roi_labels = self.proposal_target_creator(
            rois, bboxes, labels,
            self.loc_normalize_mean,
            self.loc_normalize_std
        )
        roi_sample_idx = torch.zeros(len(roi_samples))
        roi_cls_locs, roi_scores = self.faster_rcnn.head(
            features, roi_samples, roi_sample_idx
        )

        return (
            rpn_locs, rpn_scores,
            gt_rpn_locs, gt_rpn_labels,
            roi_cls_locs, roi_scores,
            gt_roi_locs, gt_roi_labels,
            rois
        )
