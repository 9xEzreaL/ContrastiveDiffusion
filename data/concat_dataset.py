import torch.utils.data as data
import albumentations as A
from torchvision import transforms
from PIL import Image
import cv2
import os
import glob
import torch
import numpy as np
import tifffile as tiff
import random
import torch.nn as nn
import torch.nn.functional as F

# from model.unet import Unet

# from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def to_8bit(img):
    img = np.array(img)
    img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


class PainDataset(data.Dataset):
    def __init__(self, data_root, eff_root, mean_root, data_len=-1, image_size=[384, 384], mode='train', mask_type="all", threshold=0.06):
        # images data root
        imgs = sorted(glob.glob(data_root))
        self.eff_root = eff_root
        self.mean_root = mean_root
        self.mask_type = mask_type
        self.threshold = threshold
        assert mask_type in ["all", "eff", "mess"], f"mask_type should be in [all, eff, mess] but got {mask_type}."
        assert len(imgs) >0, f"len of data_root({data_root}) = 0, correct data_root in config file."
        assert len(glob.glob(os.path.join(eff_root, "*"))) >0, f"len of eff_root = 0, correct eff_root in config file."
        assert len(glob.glob(os.path.join(mean_root, "*"))) >0, f"len of mean_root = 0, correct mean_root in config file."

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs[:]

        if mode == 'test':
            # ids = [i + x for i in [1219, 1242, 2691, 5589, 6900, 9246, 9338, 9522] for x in range(23)]
            ids = [i * 23 + x for i in [y for y in range(50)] for x in range(23)]
            self.imgs = [imgs[i] for i in ids]

        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
        ])
        self.image_size = image_size

        self.model = torch.load('submodels/atten_0706.pth', map_location='cpu').eval()
        # First blur kernal for effusion
        self.kernal_size = 13
        self.conv = nn.Conv2d(1,1,self.kernal_size,1,self.kernal_size//2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        # Second blur kernal for mean
        self.kernal_size_2 = 13
        self.conv_2 = nn.Conv2d(1,1,self.kernal_size_2,1,self.kernal_size_2//2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        id = path.split("/")[-1]

        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        transformed = self.tfs(image=img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        mask, one_hot_pred = self.get_mask_from_eff_mean(tiff.imread(os.path.join(self.eff_root, id)),
                                   tiff.imread(os.path.join(self.mean_root, id)),
                                   img=img
                                   )
        mask = torch.unsqueeze(mask, 0)
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        cond_image = torch.cat([cond_image, one_hot_pred], 0)
        mask_img = img * (1. - mask) + mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask_from_eff_mean(self, mask, mask_2=None, img=None):
        threshold = 0

        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if img is not None:
            img = torch.unsqueeze(img, 0)
            pred = self.model(img)
            pred = torch.argmax(pred, 1, True)
            one_hot_pred = F.one_hot(pred, 3)
            one_hot_pred = torch.permute(one_hot_pred, (0, 4, 2, 3, 1)).squeeze(0).squeeze(-1).to(torch.float)

        if self.mask_type in ["all", "mess"]:
            if isinstance(self.threshold, list):
                threshold = random.uniform(self.threshold[0], self.threshold[1])
            else:
                threshold = self.threshold
            threshold = threshold * self.kernal_size_2 * self.kernal_size_2
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            if self.mask_type == "all":
                mask += mask_2
            if self.mask_type == "mess":
                mask = mask_2 * pred[0, 0, ::]
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)


        if img is not None:
            return mask, one_hot_pred
        else:
            return mask




if __name__ == "__main__":
    train_dataset = PainDataset("/media/ziyi/Dataset/OAI_pain/full/bp/*", mask_config={"mask_mode": "hybrid"})

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True)
    for i in train_dataloader:
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        assert 0
