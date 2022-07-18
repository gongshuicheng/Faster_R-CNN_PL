import inspect
import importlib
import pickle as pkl
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.data_utils import Transform
from pytorch_metric_learning import samplers

from data.LABEL_NAMES import VOC_BBOX_LABEL_NAMES


class DInterface(pl.LightningDataModule):
    def __init__(
        self, 
        num_workers: int = 8, 
        dataset: str = 'pascal_voc',
        **kwargs
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.data_module = None
        self.kwargs = kwargs
        self.batch_size = 1  # old version is "kwargs['batch_size']"
        self.load_data_module()  # according to the "dataset"

    def setup(self, stage: Optional[str] = None):
        trans = Transform(dataset=self.dataset)
        # Assign train/val datasets for dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = self.instancialize(split="train", trans=trans)
            self.val_set = self.instancialize(split="val", trans=trans)
            print(f"Train Images : {len(self.train_set)}")
            print(f"Val Images : {len(self.val_set)}")
            
        # Assign test dataset for dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = self.instancialize(split="test", trans=trans)
            print(f"Test Images : {len(self.test_set)}")
            
        if stage == "predict" or stage is None:
            pass

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        # labels = [label for _, label in self.train_set]
        # sampler = samplers.MPerClassSampler(
        #     labels,
        #     self.batch_size / 3,
        #     batch_size=self.batch_size,
        #     length_before_new_iter=len(self.train_set)
        # )
        sampler = None
        # return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers,
        #                   shuffle=(False if sampler else True), drop_last=True, sampler=sampler)

        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        package = __package__ if __package__ else "data"

        self.data_module = getattr(
            importlib.import_module('.' + name, package=package),
            camel_name
        )

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = list(inspect.signature(self.data_module.__init__).parameters.keys())[1:]
        # class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


if __name__ == "__main__":

    data_interface = DInterface(data_dir=r'../test_data/')
    data_interface.setup(stage="fit")
    data_interface.setup(stage="test")

    from data_utils import *
    img, bboxes, labels, diffs, scale = data_interface.train_set[0]
    img = Transform.denormalize(img)
    img = T.ToPILImage()(img)
    draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
    img.show()
    img, bboxes, labels, diffs, scale = data_interface.val_set[0]
    img = Transform.denormalize(img)
    img = T.ToPILImage()(img)
    draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
    img.show()
    img, bboxes, labels, diffs, scale = data_interface.test_set[0]
    img = Transform.denormalize(img)
    img = T.ToPILImage()(img)
    draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
    img.show()
