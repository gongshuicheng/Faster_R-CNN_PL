import inspect
import importlib

import torch
from torch.nn import functional as F

# from torchsummary import summary

import pytorch_lightning as pl
from model.loss.faster_rcnn_loss import fast_rcnn_loc_loss, smooth_l1_loss
from model.utils.bbox_tools import loc2bbox
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        # Parser
        self.save_hyperparameters()

        # Model
        self.model = None
        self.model_module = None
        self.load_model()
        
        # Device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.to(device)

        # Loss
        self.rpn_loc_loss_func = None
        self.rpn_cls_loss_func = None
        self.roi_loc_loss_func = None
        self.roi_cls_loss_func = None
        self.configure_loss()

        # Metrics
        self.train_metrics = MeanAveragePrecision()
        self.val_metrics = MeanAveragePrecision()

        # Debug
        # torch.autograd.set_detect_anomaly(True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
    
        imgs, bboxes, labels, diffs, scale = batch  # all on cpu
        
        (
            rpn_locs, rpn_scores,
            gt_rpn_locs, gt_rpn_labels,
            roi_cls_locs, roi_scores,
            gt_roi_locs, gt_roi_labels,
            roi_samples
        ) \
            = self(imgs, bboxes, labels, scale)

        # RPN LOSS
        rpn_loc_loss = self.rpn_loc_loss_func(
            rpn_locs, gt_rpn_locs, gt_rpn_labels, self.model.rpn_sigma
        )
        rpn_cls_loss = self.rpn_cls_loss_func(rpn_scores, gt_rpn_labels.long().cuda(), ignore_index=-1)

        # ROI LOSS TODO: To understand!
        n_sample = roi_cls_locs.shape[0]

        roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)
        roi_locs = roi_cls_locs[torch.arange(0, n_sample), gt_roi_labels.long()]
        roi_loc_loss = self.roi_loc_loss_func(
            roi_locs, gt_roi_locs, gt_roi_labels, self.model.roi_sigma
        )
        roi_cls_loss = self.roi_cls_loss_func(roi_scores, gt_roi_labels.long())

        # TOTAL LOSS
        loss = torch.sum(rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss)

        self.log("t_rpn_loc_loss", rpn_loc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("t_rpn_cls_loss", rpn_cls_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("t_roi_loc_loss", roi_loc_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("t_roi_cls_loss", roi_cls_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("t_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # print("Model is :", self.model.training)  # False
        
        imgs, bboxes, labels, diffs, scale = batch  # All on cpu with in one-batch
        
        
        # pred_bboxes: torch.Size([n_pred_bboxes, 4])
        # pred_labels: torch.Size([n_pred_bboxes])
        # pred_scores: torch.Size([n_pred_bboxes])
        pred_bboxes, pred_labels, pred_scores = self.model.predict(imgs, scale)
        


#         # ROI LOSS TODO: To understand!
#         n_sample = roi_cls_locs.shape[0]
#         roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)
#         roi_locs = roi_cls_locs[torch.arange(0, n_sample), gt_roi_labels.long()]
#         roi_loc_loss = self.roi_loc_loss_func(
#             roi_locs, gt_roi_locs, gt_roi_labels, self.model.roi_sigma
#         )
#         roi_cls_loss = self.roi_cls_loss_func(roi_scores, gt_roi_labels.long())

#         # TOTAL LOSS
#         loss = torch.sum(rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss)

#         roi_bboxes = loc2bbox(roi_samples, roi_locs)

        preds = [
            dict(
                boxes=pred_bboxes[..., [1, 0, 3, 2]],
                scores=pred_scores,
                labels=pred_labels,
            )
        ]
        target = [
            dict(
                boxes=bboxes[0][..., [1, 0, 3, 2]],
                labels=labels[0],
            )
        ]
        
        self.val_metrics.update(preds, target)
        # self.log("v_roi_loc_loss", roi_loc_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("v_roi_cls_loss", roi_cls_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("v_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        # return self.validation_step(batch, batch_idx)
        pass

    def training_epoch_end(self, training_step_outputs):
        self.train_metrics.reset()

    def validation_epoch_end(self, val_step_outputs):
        self.log("v_mAP", self.val_metrics.compute(), on_epoch=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        # Init
        params = []
        optimizer = None
        scheduler = None
 
        # Model Parameters
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': self.hparams.lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay}]
        
        # Optimizer            
        if self.hparams.optim == "adam":
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
            optimizer = torch.optim.Adam(params)
        elif self.hparams.optim == "sgd":
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
            optimizer = torch.optim.SGD(params, momentum=0.9)
        
        # LR Scheduler
        if self.hparams.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1, verbose=True)
        elif self.hparams.lr_scheduler == "cosine":
            pass
        else:
            return optimizer
        
        return ([optimizer], [scheduler])

    def configure_loss(self):
        self.rpn_loc_loss_func = fast_rcnn_loc_loss
        self.rpn_cls_loss_func = F.cross_entropy
        self.roi_loc_loss_func = fast_rcnn_loc_loss
        self.roi_cls_loss_func = F.cross_entropy

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        # class_args = inspect.getargspec(Model.__init__).args[1:]
        class_args = list(inspect.signature(self.model_module.__init__).parameters.keys())[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)