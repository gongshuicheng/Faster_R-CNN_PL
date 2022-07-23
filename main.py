""" This main entrance of the whole project.
    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""

import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer

import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger


from model import MInterface
from data import DInterface

from utils import load_model_path_by_args


# def load_callbacks():
#     callbacks = []
#     callbacks.append(plc.EarlyStopping(
#         monitor='val_acc_epoch',
#         mode='max',
#         patience=10,
#         min_delta=0.001
#     ))
#
#     callbacks.append(plc.ModelCheckpoint(
#         monitor='val_acc_epoch',
#         filename='best-{epoch:02d}-{val_acc:.3f}',
#         save_top_k=1,
#         mode='max',
#         save_last=True
#     ))
#
#     if args.lr_scheduler:
#         callbacks.append(plc.LearningRateMonitor(
#             logging_interval='epoch'))
#     return callbacks


def main(args):
    # Set the random seed!
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    # args.callbacks = load_callbacks()
    # args.logger = logger
    
    # args.callbacks = load_callbacks()
    # args.logger = TensorBoardLogger(
    #     'kfold_log', 
    #     name=args.log_dir,
    # )
    

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Dataset
    parser.add_argument("--year", default=["2007", "2012"], type=list)
    
    # Basic Training Control
    # parser.add_argument("--val_mode", default="three-way-holdout", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # LR Scheduler
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument("--optim", default="sgd", type=str)
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    # parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    # parser.add_argument('--lr_decay_min_lr', default=1e-3, type=float)
    # parser.add_argument("--lr_gamma", default=0.99, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset_name', default='pascal_voc', type=str)
    parser.add_argument('--data_dir', default='./test_data', type=str)
    parser.add_argument('--model_name', default='faster_rcnn_vgg16', type=str)
    # parser.add_argument('--loss', default='fast_rcnn_loss', type=str)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--in_channel', default=3, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(accelerator="gpu", device=1)
    parser.set_defaults(max_epochs=14)
    parser.set_defaults(log_every_n_steps=100)
    
    args = parser.parse_args()

    # List Arguments
    # args.mean_sen = [0.485, 0.456, 0.406]
    # args.std_sen = [0.229, 0.224, 0.225]

    main(args)