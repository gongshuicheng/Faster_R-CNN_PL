import inspect
import importlib

import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torchsummary import summary

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

        # Loss
        self.rpn_loc_loss_func = None
        self.rpn_cls_loss_func = None
        self.roi_loc_loss_func = None
        self.roi_cls_loss_func = None
        self.configure_loss()

        # Metrics
        self.train_metrics = MeanAveragePrecision()
        self.val_metrics = MeanAveragePrecision()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        imgs, bboxes, labels, diffs, scale = batch
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
        rpn_cls_loss = self.rpn_cls_loss_func(rpn_scores, gt_rpn_labels.long(), ignore_index=-1)

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

        # loss = sum(loss)

        roi_bboxes = loc2bbox(roi_samples, roi_locs)

        preds = [
            dict(
                boxes=roi_bboxes,
                scores=roi_scores,
                labels=torch.argmax(F.softmax(rpn_scores)),
            )
        ]
        target = [
            dict(
                boxes=bboxes,
                labels=gt_roi_labels,
            )
        ]
        # self.train_metrics.update(preds, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # data, targets = batch
        # d, p = self(data)
        # loss = self.loss_function(d, p, targets)
        # preds = p.argmax(axis=1)
        #
        # self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # return {"loss": loss, "preds": preds, "targets": targets}
        pass

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        # return self.validation_step(batch, batch_idx)
        pass

    def training_epoch_end(self, training_step_outputs):
        # preds = []
        # targets = []
        # for out in training_step_outputs:
        #     preds.append(out["preds"])
        #     targets.append(out["targets"])
        # preds = torch.cat(preds)
        # targets = torch.cat(targets)
        self.train_metrics.reset()

    def validation_epoch_end(self, val_step_outputs):
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = None
        # params = []
        # for key, value in dict(self.model.named_parameters()).items():
        #     if value.requires_grad:
        #         if 'bias' in key:
        #             params += [{'params': [value], 'lr': self.hparams.lr * 2, 'weight_decay': 0}]
        #         else:
        #             params += [{'params': [value], 'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay}]

        if self.hparams.optim == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optim == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)

        return optimizer

        # if self.hparams.lr_scheduler is None:
        #     return optimizer

        # else:
        #     if self.hparams.lr_scheduler == 'step':
        #         scheduler = lrs.StepLR(
        #             optimizer,
        #             step_size=self.hparams.lr_decay_steps,
        #             gamma=self.hparams.lr_decay_rate
        #         )
        #     elif self.hparams.lr_scheduler == 'cosine':
        #         scheduler = lrs.CosineAnnealingLR(
        #             optimizer,
        #             T_max=self.hparams.lr_decay_steps,
        #             eta_min=self.hparams.lr_decay_min_lr
        #         )
        #     elif self.hparams.lr_scheduler == "exp":
        #         scheduler = lrs.ExponentialLR(
        #             optimizer,
        #             gamma=self.hparams.lr_gamma,
        #         )
        #     else:
        #         raise ValueError('Invalid lr_scheduler type!')
        #     return [optimizer], [scheduler]

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
        # summary(self.model, torch.zeros((1, 3, 224, 224)), torch.zeros((1, 4)), torch.zeros((1,)), 1)

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