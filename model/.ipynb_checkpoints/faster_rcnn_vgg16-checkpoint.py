import torch
import torch.nn as nn

from torchvision.ops import RoIPool
from torchvision.models import vgg16

from model.faster_rcnn import FasterRcnn, RegionProposalNetwork
from model.utils.init_tools import init_layer


def decom_vgg16(pretrained=True, use_drop=True):
    model = vgg16(pretrained=pretrained)
    extractor = model.features[:30]
    classifier = list(model.classifier)
    del classifier[6]
    if not use_drop:
        del classifier[5]
    classifier = nn.Sequential(*classifier)

    # Freeze top4 Conv.
    for layer in extractor[:10]:
        for p in layer.parameters():
            p.require_grad = False

    return extractor, classifier  # Feature extractor and Classifier


class Vgg16RoiHead(nn.Module):
    def __init__(
            self,
            n_classes,
            roi_size, spatial_scale,
            classifier
    ):
        # n_class includes the background
        super(Vgg16RoiHead, self).__init__()

        self.classifier = classifier.cuda()  # classifier is on cpu
        self.cls_locs = nn.Linear(4096, n_classes * 4)
        self.scores = nn.Linear(4096, n_classes)

        init_layer(self.cls_locs, 0, 0.001)
        init_layer(self.scores, 0, 0.01)

        # RoIPool
        self.n_classes = n_classes
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.RoIs = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_idx):
        x = x.cuda()  # x is on cpu
        rois = rois.cuda()
        roi_idx = roi_idx.cuda()
        
        idx_and_rois = torch.cat([roi_idx[:, None], rois], dim=1)
        # print("idx_and_rois : ", idx_and_rois.device)
        # print("x:", x.device)
        
        # NOTE: important: yx->xy
        xy_idx_and_rois = idx_and_rois[:, [0, 2, 1, 4, 3]]

        pool = self.RoIs(x, xy_idx_and_rois)  # pool is on cuda
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        # print("fc7:", fc7.device)
        
        roi_cls_locs = self.cls_locs(fc7)
        roi_scores = self.scores(fc7)
        
        # print("roi_cls_locs shape:", roi_cls_locs.shape)
        # print("roi_scores shape:", roi_scores.shape)
        
        
        return roi_cls_locs, roi_scores


class FasterRcnnVgg16(FasterRcnn):
    def __init__(
            self,
            n_fg_classes=20,
            ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            feat_stride=16,
            rpn_sigma=3., roi_sigma=1
    ):
        self.__dict__.update(locals())

        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
        )
        head = Vgg16RoiHead(
            n_classes=n_fg_classes + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRcnnVgg16, self).__init__(
            extractor,
            rpn,
            head,
        )




