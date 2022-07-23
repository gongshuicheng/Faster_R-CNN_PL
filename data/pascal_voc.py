import os
import warnings
import xml.etree.ElementTree as xmlET

from torch.utils.data import Dataset

from data.data_utils import *
from data.LABEL_NAMES import VOC_BBOX_LABEL_NAMES


class PascalVoc(Dataset):
    """
    Pascal VOC :

    Args:
        data_dir (string): Path to the root of the training data
        split ({"train", "val", "trainval", "test"}): splits of the datasets


    """
    def __init__(
        self,
        data_dir=r'../Datasets/PASCALVOC/VOCdevkit/VOC2007',
        year=["2007", "2012"],
        trans=None,
        split="train",
        use_difficult=True,
        return_difficult=True,
        label_name=VOC_BBOX_LABEL_NAMES,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.ids = []
        self.check_files()

    def check_files(self):
        if self.split not in ["trainval", "train", "val", "test"]:
            warnings.warn(
                'Please pick split from \'train\', \'trainval\', \'val\''
                'for 2012 dataset. For 2007 dataset, pick \'test\''
                ' in addition to the above mentioned splits.'
            )
        id_list_file = os.path.join(
            self.data_dir, f"ImageSets/Main/{self.split}.txt"
        )
        self.ids = [id_.strip() for id_ in open(id_list_file)]

    def __len__(self):
        return len(self.ids)

    def get_annotations(self, id_):
        annot = xmlET.parse(
            os.path.join(self.data_dir, "Annotations", id_ + ".xml")
        )

        bboxes_list = []
        labels_list = []
        difficulties_list = []

        for obj in annot.findall("object"):
            if not self.use_difficult and int(obj.find("difficult").text) == 1:
                continue
            difficulties_list.append(int(obj.find("difficult").text))
            bbox_anno = obj.find("bndbox")
            bboxes_list.append([
                int(bbox_anno.find(tag).text) - 1
                for tag in ("ymin", "xmin", "ymax", "xmax")
            ])
            name = obj.find("name").text.lower().strip()
            labels_list.append(self.label_name.index(name))

        bboxes = torch.Tensor(bboxes_list).type(torch.float32)
        labels = torch.Tensor(labels_list).type(torch.int32)
        difficulties = torch.Tensor(difficulties_list).type(torch.uint8)

        return bboxes, labels, difficulties

    def __getitem__(self, idx):
        """
        Returns the i-th example

        Args:
            idx: The index of the example

        Returns:
            tuple of an image and its bounding boxes
        """
        id_ = self.ids[idx]

        img_path = os.path.join(self.data_dir, "JPEGImages", id_ + ".jpg")
        img = Image.open(img_path)  # PIL type
        bboxes, labels, difficulties = self.get_annotations(id_)
        scale = torch.tensor([1.0], dtype=torch.float32)

        if self.trans:
            img, bboxes, labels, difficulties, scale = self.trans(
                img, bboxes, labels, difficulties, split=self.split
            )

        return img, bboxes, labels, difficulties, scale


if __name__ == "__main__":
    # test_method = "fiftyone"
    test_method = "dataset_class"

    if test_method == "fiftyone":
        # Test the test data
        # import fiftyone as fo
        # name = "test_data"
        # data_path = "../test_data/JPEGImages"
        # labels_path = "../test_data/Annotations"
        #
        # try:
        #     dataset = fo.Dataset.from_dir(
        #         dataset_type=fo.types.VOCDetectionDataset,
        #         data_path=data_path,
        #         labels_path=labels_path,
        #         name=name
        #     )
        # except:
        #     dataset = fo.load_dataset(name)
        # session = fo.launch_app(dataset)
        input("Press Enter to end...")

    elif test_method == "process":
        process_method = "flip"
        dataset = PascalVoc(data_dir="../test_data/", trans=None)
        img, bboxes, labels, difficulties, scale = dataset[0]
        # Process Image
        if process_method == "origin":
            draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
        elif process_method == "resize":
            img, params = resize_img(img, short_max=600., long_max=1000.)  # Resize
            bboxes = resize_bbox(bboxes, params)
            draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
        elif process_method == "flip":
            img, params = flip_img(img, p=1)
            bboxes = flip_bbox(bboxes, params)
            draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
        elif process_method == "crop":
            img, params = crop_img(img, input_dim=224)
            params["allow_outside_center"] = False
            bboxes, mask_idx = crop_bbox(bboxes, params)
            labels = labels[mask_idx]
            draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
        img.show()

    elif test_method == "dataset_class":
        trans = Transform()
        dataset = PascalVoc(data_dir="../test_data/", trans=trans, split="train")
        img, bboxes, labels, difficulties, scale = dataset[0]
        img = Transform.denormalize(img)
        img = T.ToPILImage()(img)
        draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
        img.show()











