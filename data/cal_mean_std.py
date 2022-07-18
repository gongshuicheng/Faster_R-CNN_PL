import os
from torchvision import transforms as T
from PIL import Image
import torch


def cal_mean_std(data_dir, input_dim):
    means, std = [], []
    start = 0

    for i, img_name in enumerate(os.listdir(data_dir)):
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):

            # Test
            # print(trans(Image.open(os.path.join(data_dir, img_name))))

            img_path = os.path.join(data_dir, img_name)
            img = Image.open(img_path)

            trans = T.Compose([
                T.RandomCrop(input_dim),
                T.ToTensor()
            ])
            img = trans(img)
            img = img.unsqueeze(-1)
            if start == 0:
                imgs = img
                start = 1

            imgs = torch.cat((imgs, img), axis=3)

    for i in range(3):
        pixels = imgs[i, :, :, :].flatten(1)
        means.append(torch.mean(pixels))
        std.append(torch.std(pixels))

    return torch.Tensor(means), torch.Tensor(std)


if __name__ == "__main__":
    data_dir = "../test_data/JPEGImages"
    mean, std = cal_mean_std(data_dir, input_dim=224)
    print(mean, std)