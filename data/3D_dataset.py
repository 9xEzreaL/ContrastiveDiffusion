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
    def __init__(self, data_root, eff_root, mean_root, data_len=-1, image_size=[384, 384], mode='train', mask_type="all"):
        # images data root
        imgs = sorted(glob.glob(data_root))
        self.eff_root = eff_root
        self.mean_root = mean_root
        self.mask_type = mask_type
        assert len(imgs) >0, f"len of data_root({data_root}) = 0, correct data_root in config file."
        assert len(glob.glob(os.path.join(eff_root, "*"))) >0, f"len of eff_root = 0, correct eff_root in config file."
        assert len(glob.glob(os.path.join(mean_root, "*"))) >0, f"len of mean_root = 0, correct mean_root in config file."

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs[:]

        if mode == 'test':
            ids = [i + x for i in [1219, 1242, 2691, 5589, 6900, 9246, 9338, 9522] for x in range(23)]
            # ids = [i * 23 + x for i in [y for y in range(50)] for x in range(23)]
            self.imgs = [imgs[i] for i in ids]

        self.num_subject = list(set([ID.rsplit("_", 1)[0] for ID in self.imgs]))

        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
        ])
        self.image_size = image_size

        self.model = torch.load('submodels/80.pth', map_location='cpu').eval()
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
        path = self.num_subject[index]
        # path = self.imgs[index]
        id = path.split("/")[-1]
        ###############
        img_cube, mask_cube, cond_cube, mask_img_cube = self.get_3D_from_ID(path)


        ret['gt_image'] = img_cube
        ret['cond_image'] = cond_cube
        ret['mask_image'] = mask_img_cube
        ret['mask'] = mask_cube
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        print(len(self.num_subject))
        return len(self.num_subject)
        # return len(self.imgs)

    def get_3D_from_ID(self, path):
        id = path.rsplit("/")[-1]
        img_cube = []
        mask_cube = []
        cond_img_cube = []
        mask_img_cude_for_visualize = []
        for i in range(23):
            img = tiff.imread(path + "_" + str(i).zfill(3) + ".tif")
            img = (img - img.min()) / (img.max() - img.min())
            img = (img - 0.5) / 0.5
            img = np.expand_dims(img, 0)
            img_cube.append(img)

            mask = self.get_mask_from_eff_mean(tiff.imread(os.path.join(self.eff_root, id + "_" + str(i).zfill(3) + ".tif")),
                                             tiff.imread(os.path.join(self.mean_root, id + "_" + str(i).zfill(3) + ".tif")),
                                             img=img
                                             )
            mask = np.expand_dims(mask, 0)
            mask_cube.append(mask)

            cond_image = img * (1. - mask) + mask * np.array(torch.randn_like(torch.from_numpy(img)))
            cond_img_cube.append(cond_image)

            mask_img = img * (1. - mask) + mask
            mask_img_cude_for_visualize.append(mask_img)

        img_cube = np.expand_dims(np.concatenate(img_cube, 0), 0).astype(np.float32)
        mask_cube = np.expand_dims(np.concatenate(mask_cube, 0), 0).astype(np.float32)
        cond_img_cube = np.expand_dims(np.concatenate(cond_img_cube, 0), 0).astype(np.float32)
        mask_img_cude_for_visualize = np.expand_dims(np.concatenate(mask_img_cude_for_visualize, 0), 0).astype(np.float32)
        return img_cube, mask_cube, cond_img_cube, mask_img_cude_for_visualize

    def get_mask_from_eff_mean(self, mask, mask_2=None, img=None):
        threshold = 0

        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if img is not None:
            img = torch.unsqueeze(torch.from_numpy(img).float(), 0)
            pred = self.model(img)
            pred = np.array(torch.argmax(pred, 1, True))

        if self.mask_type in ["all", "mess"]:
            threshold = 0.06 * self.kernal_size_2 * self.kernal_size_2
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            if self.mask_type == "all":
                mask += mask_2
            if self.mask_type == "mess":
                mask = mask_2 * pred[0, 0, ::]
                # mask = mask_2 - mask
            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

        return mask




if __name__ == "__main__":
    train_dataset = PainDataset(data_root="/media/ExtHDD01/Dataset/OAI_pain/full/full/bp/*",
                                eff_root="/media/ExtHDD01/Dataset/OAI_pain/full/full/apeff",
                                mean_root="/media/ExtHDD01/Dataset/OAI_pain/full/full/apmean_102323")

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True)
    for i in train_dataloader:
        print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        print(i["cond_image"].shape)
        print(i["mask_image"].shape)
        assert 0
