import numpy as np

import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw

from data.LABEL_NAMES import VOC_BBOX_LABEL_NAMES


def resize_img(img, short_max=600., long_max=1000.):
    params = {}
    W, H = img.size
    if H <= W:
        y_scale = short_max / H
        x_scale = long_max / W
        scale = min(y_scale, x_scale)
        img = T.Resize(int(H * scale + 0.5))(img)
    else:
        y_scale = long_max / H
        x_scale = short_max / W
        scale = min(y_scale, x_scale)
        img = T.Resize(int(W * scale + 0.5))(img)
    params["scale"] = scale
    return img, params


def resize_bbox(bbox, params):
    return bbox * params["scale"]


def flip_img(img, p=0.5):
    params = {}
    params["size_flip"] = img.size
    params["y_flip"] = np.random.choice([True, False], p=[p, 1 - p]),
    params["x_flip"] = np.random.choice([True, False], p=[p, 1 - p])
    if params["y_flip"]:
        img = T.RandomVerticalFlip(1)(img)
    if params["x_flip"]:
        img = T.RandomHorizontalFlip(1)(img)
    return img, params


def flip_bbox(bboxes, params):
    W, H = params["size_flip"]
    if params["y_flip"]:
        y_max = H - bboxes[:, 0]
        y_min = H - bboxes[:, 2]
        bboxes[:, 0] = y_min
        bboxes[:, 2] = y_max
    if params["x_flip"]:
        x_max = W - bboxes[:, 1]
        x_min = W - bboxes[:, 3]
        bboxes[:, 1] = x_min
        bboxes[:, 3] = x_max
    return bboxes


def crop_img(img, input_dim=224):
    if type(input_dim) == int:
        input_dim = [input_dim, input_dim]
    elif len(input_dim) == 1 and type(input_dim) == list:
        input_dim.extend(input_dim)
    elif len(input_dim) == 1 and type(input_dim) == tuple:
        input_dim = input_dim + input_dim
    params = {}
    W, H = img.size
    params["y_slice_start"] = np.random.randint(0, H - input_dim[0] + 1)
    params["y_slice_end"] = params["y_slice_start"] + input_dim[0]
    params["x_slice_start"] = np.random.randint(0, W - input_dim[1] + 1)
    params["x_slice_end"] = params["x_slice_start"] + input_dim[1]
    img = img.crop((
        params["x_slice_start"], params["y_slice_start"],
        params["x_slice_end"], params["y_slice_end"]
    ))
    return img, params


def crop_bbox(bboxes, params):
    """

    Args:
        bboxes (list torch.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        params (dict): Dictionary of the params

    Returns:
        bboxes (list torch.ndarray):
    """
    # print(bboxes)
    crop_bd = torch.tensor((
        params["y_slice_start"],
        params["x_slice_start"],
        params["y_slice_end"],
        params["x_slice_end"],
    ))
    # print(crop_bd)

    if "allow_outsize_center" not in params:  # Set False as default
        params["allow_outside_center"] = False
    if params["allow_outside_center"]:
        mask = torch.ones(bboxes.shape[0], dtype=torch.bool)
    else:
        center = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0  # (cy, cx)
        mask = torch.all(
            torch.logical_and(
                crop_bd[:2] <= center,
                center < crop_bd[2:]
            ),
            1
        )

    bboxes[:, :2] = torch.maximum(bboxes[:, :2], crop_bd[:2])
    bboxes[:, 2:] = torch.minimum(bboxes[:, 2:], crop_bd[2:] - 1)
    bboxes[:, :2] -= crop_bd[:2]
    bboxes[:, 2:] -= crop_bd[:2]

    # Validate
    mask = torch.logical_and(
        mask,
        torch.all(bboxes[:, :2] < bboxes[:, 2:], axis=1)
    )
    return bboxes[mask, ...], torch.nonzero(mask)


def normalize(img, mean=None, std=None):
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    img = trans(img)
    return img


class Transform(object):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __init__(
            self,
            input_dim=224,
            aug=False, p=0.5,
            dataset="pascal_voc",
    ):
        self.__dict__.update(locals())
        if self.dataset == "pascal_voc":
            self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(
            self,
            img, bboxes, labels, difficulties, split="train"
    ):
        params_log = {}
        # Resize
        img, params = resize_img(img)
        bboxes = resize_bbox(bboxes, params)
        params_log.update(params)

        if self.aug:  # TODO: Add other augment methods
            # Random Crop
            img, params = crop_img(img, input_dim=self.input_dim)
            params["allow_outside_center"] = False
            bboxes, mask_idx = crop_bbox(bboxes, params)
            params["mask_idx"] = mask_idx
            labels = labels[mask_idx]
            diff_list = []
            for i in range(mask_idx.shape[0]):
                if torch.any(difficulties == mask_idx[i]):
                    diff_list.append(mask_idx[i])
            difficulties = torch.Tensor(diff_list).type(torch.uint8)
            params_log.update(params)
            # Random Flip
            img, params = flip_img(img, p=self.p)
            bboxes = flip_bbox(bboxes, params)
            params_log.update(params)

        img = normalize(img, mean=self.mean, std=self.std)

        return img, bboxes, labels, difficulties, params_log["scale"]

    @classmethod
    def denormalize(cls, img):
        C, H, W = img.shape
        mean = cls.mean
        std = cls.std
        img_copy = img.view(3, -1)
        img_copy = torch.clamp((img_copy * std.view(3, -1) + mean.view(3, -1)), 0, 1)
        return img_copy.view(3, H, W)


def preprocess(img):  # Input PIL image
    img = resize_img(img)
    img = normalize(img)
    return img


def draw_bboxes_labels(img, bboxes, labels, label_name):
    draw = ImageDraw.Draw(img)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        draw.rectangle(
            # [(xmin, ymin), (xmax, ymax)]
            [(int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2]))],
            outline="red"
        )
        draw.text(
            (int(bbox[1]), int(bbox[0])),
            label_name[labels[i]],
            fill="red",
            align="right"
        )


if __name__ == "__main__":
    img_path = "../test_data/JPEGImages/000005.jpg"

    from PIL import Image
    img = Image.open(img_path)
    # annot_path = "../test_data/Annotations/000005.xml"
    bboxes = torch.Tensor([
        [210., 262., 338., 323.],
        [263., 164., 371., 252.],
        [243.,   4., 373.,  66.],
        [193., 240., 298., 294.],
        [185., 276., 219., 311.]
    ]).type(torch.float32)
    labels = torch.tensor([8, 8, 8, 8, 8]).type(torch.int32)
    difficulties = torch.tensor([3]).type(torch.uint8)
    trans = Transform(dataset="pacal_voc")
    img, bboxes, labels, difficulties, scale = trans(img, bboxes, labels, difficulties)
    print(difficulties)
    img = Transform.denormalize(img)
    img = T.ToPILImage()(img)
    draw_bboxes_labels(img, bboxes, labels, VOC_BBOX_LABEL_NAMES)
    img.show()



