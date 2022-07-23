import torch


from model.utils.bbox_tools import bbox_iou, bbox2loc
from model.utils.no_grad import without_grad

class AnchorTargetCreator(object):
    def __init__(
            self,
            n_samples=256,
            pos_ratio=0.5,
            pos_iou_thresh=0.7, neg_iou_thresh=0.3,
    ):
        self.n_samples = n_samples
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
    
    @without_grad
    def __call__(self, gt_bboxes, shifted_anchors, img_size):
        img_h, img_w = img_size[0], img_size[1]
        n_anchors = len(shifted_anchors)
        
        # gt_bboxes with the shape of (1, n_gt_bboxes, 4)

        # Get inside index
        inside_idx = torch.where(
            (shifted_anchors[:, 0] >= 0) &
            (shifted_anchors[:, 1] >= 0) &
            (shifted_anchors[:, 2] <= img_h) &
            (shifted_anchors[:, 3] <= img_w)
        )[0]  # [0] means getting the row index of where
        inside_anchors = shifted_anchors[inside_idx]

        # Get the Max iou bounding box for anchors and ground truth respectively
        ious = bbox_iou(inside_anchors, gt_bboxes[0])  # (n_cells * n_anchors) x N_gt
        anchor_argmax_ious = ious.argmax(axis=1)  # Argmax for each anchor
        anchor_max_ious = ious[torch.arange(ious.shape[0]), anchor_argmax_ious]
        gt_bbox_argmax_ious = ious.argmax(axis=0)  # Argmax for each ground truth
        gt_bbox_max_ious = ious[gt_bbox_argmax_ious, torch.arange(ious.shape[1])]
        gt_bbox_argmax_ious = torch.where(ious == gt_bbox_max_ious)[0]
        # Get the row num of all the "True" of equal max value

        # Create labels for anchors
        # label: 1 = positive; label: 0 = negative; label: -1 = ignore
        created_labels = - torch.ones((len(inside_idx),), dtype=torch.float32)  # Fill the labels with -1
        created_labels[anchor_max_ious < self.neg_iou_thresh] = 0  # 0 for the max iou smaller than neg thresh
        created_labels[gt_bbox_argmax_ious] = 1  # 1 for the max iou with ground truth
        created_labels[anchor_max_ious >= self.pos_iou_thresh] = 1  # 1 for the max iou larger than pos thresh

        n_pos = int(self.pos_ratio * self.n_samples)
        pos_idx = torch.where(created_labels == 1)[0]  #
        if len(pos_idx) > n_pos:  # when the num of pos items is larger than it should be
            n_pos_idx = len(pos_idx)
            disable_idx = pos_idx[torch.randperm(n_pos_idx)[:n_pos_idx - n_pos]]
            created_labels[disable_idx] = -1

        n_neg = self.n_samples - torch.sum(created_labels == 1)
        neg_idx = torch.where(created_labels == 0)[0]
        if len(neg_idx) > n_neg:
            n_neg_idx = len(neg_idx)
            disable_idx = neg_idx[torch.randperm(n_neg_idx)[:n_neg_idx - n_neg]]
            created_labels[disable_idx] = -1

        # Compute the ground true bounding boxes' regression
        gt_locs = bbox2loc(inside_anchors, gt_bboxes[0][anchor_argmax_ious])

        # Return to the original size
        unmapped_created_labels = - torch.ones((n_anchors,), dtype=created_labels.dtype)
        unmapped_created_labels[inside_idx] = created_labels
        unmapped_gt_locs = torch.zeros((n_anchors, gt_locs.shape[1]), dtype=gt_locs.dtype)
        unmapped_gt_locs[inside_idx] = gt_locs

        # print(f"Total {len(torch.where(unmapped_created_labels == 1)[0])} positive labels!")
        # print(f"Total {len(torch.where(unmapped_created_labels == 0)[0])} negative labels!")
        # print(f"Total {len(torch.where(unmapped_created_labels == -1)[0])} ignored labels!")

        return unmapped_gt_locs, unmapped_created_labels


class ProposalTargetCreator(object):
    def __init__(
            self,
            n_samples=128,
            pos_ratio=0.25,
            pos_iou_thresh=0.5,
            neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
    ):
        self.n_samples = n_samples
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
    
    @without_grad
    def __call__(
            self,
            rois, gt_bboxes, gt_labels,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
    ):
        device = rois.device

        
        # rois =  # TODO: To Understand
        n_bboxes = gt_bboxes.shape[1]
        n_rois = rois.shape[0]
        # print(n_bboxes, n_rois)

        ious = bbox_iou(rois, gt_bboxes[0])
        roi_argmax_ious = ious.argmax(axis=1)
        roi_max_ious = ious[torch.arange(n_rois), roi_argmax_ious]
        gt_bbox_argmax_ious = ious.argmax(axis=0)
        gt_bbox_max_ious = ious[gt_bbox_argmax_ious, torch.arange(n_bboxes)]

        # Select foreground rois >= pos_iou_thresh iou.
        n_pos_rois = int(self.n_samples * self.pos_ratio)
        pos_idx = torch.where(roi_max_ious >= self.pos_iou_thresh)[0]
        if n_pos_rois < len(pos_idx):
            pos_idx = pos_idx[torch.randperm(len(pos_idx))[:n_pos_rois]]

        # Select background rois in [neg_iou_thresh_lo, neg_iou_thresh_hi).
        n_neg_rois = self.n_samples - len(pos_idx)
        neg_idx = torch.where(
            (roi_max_ious < self.neg_iou_thresh_hi) &
            (roi_max_ious >= self.neg_iou_thresh_lo)
        )[0]
        if n_pos_rois < len(neg_idx):
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg_rois]]

        keep_idx = torch.cat((pos_idx, neg_idx), dim=0)
        # print(f"Total {keep_idx.shape[0]} sampled rois.")
        # print(f"Total {len(pos_idx)} positive rois (with labels).")

        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        
        # gt_labels: torch.Size([1, n_bboxes])
        # roi_argmax_ious: torch.Size([n_rois])
        # gt_roi_labels: torch.Size([n_rois]) -> torch.Size([n_samples])
        # roi_samples: torch.Size([n_samples, 4])
        gt_roi_labels = gt_labels[0][roi_argmax_ious] + 1
        gt_roi_labels = gt_roi_labels[keep_idx]
        gt_roi_labels[len(pos_idx):] = 0  # negative labels --> 0
        roi_samples = rois[keep_idx]

        # Compute offsets and scales to match sampled rois to the GTs.
        roi_samples = roi_samples.to(device)
        gt_roi_locs = bbox2loc(roi_samples, gt_bboxes[0][roi_argmax_ious[keep_idx]])
        
        # Normalize
        gt_roi_locs = (gt_roi_locs.to(device) - torch.tensor(loc_normalize_mean, dtype=torch.float32).to(device)) \
                       / torch.tensor(loc_normalize_std, dtype=torch.float32).to(device)

        return roi_samples, gt_roi_locs, gt_roi_labels

