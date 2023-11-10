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
from typing import List

# from model.unet import Unet

# from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)


def pil_loader(path):
    return Image.open(path).convert('RGB')


def to_8bit(img):
    img = np.array(img)
    img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


class PainDataset(data.Dataset):
    def __init__(self, data_root, eff_root, mean_root, data_len=-1, image_size=[384, 384], mode='train'):
        ap_imgs = sorted(glob.glob(data_root))
        self.eff_root = eff_root
        self.mean_root = mean_root
        assert len(ap_imgs) >0, f"len of data_root({data_root}) = 0, correct data_root in config file."
        assert len(glob.glob(os.path.join(eff_root, "*"))) >0, f"len of eff_root = 0, correct eff_root in config file."
        assert len(glob.glob(os.path.join(mean_root, "*"))) >0, f"len of mean_root = 0, correct mean_root in config file."

        if mode == 'train':
            # excluded id for eval(only training on bp)
            excluded_id = [i + x for i in [1219, 1242, 2691, 5589, 6900, 9246, 9338, 9522] for x in range(23)] # selected 8 case
            excluded_id_50 = [i + x for i in range(50) for x in range(23)] # first 50 case
            excluded_id = excluded_id + excluded_id_50
            imgs = [i for num, i in enumerate(ap_imgs) if num not in excluded_id]
        elif mode == 'test':
            # for ap test
            imgs = ap_imgs
        else:
            excluded_id = [i + x for i in [1219, 1242, 2691, 5589, 6900, 9246, 9338, 9522] for x in range(23)]
            ap_imgs = [i for num, i in enumerate(ap_imgs) if num in excluded_id]
            imgs = ap_imgs

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs[:]

        if mode == 'test':
            ids = [i + x for i in [1219, 1242, 2691, 5589, 6900, 9246, 9338, 9522] for x in range(23)]
            # ids = [i * 23 + x for i in [y for y in range(40, 50)] for x in range(23)]
            self.imgs = [imgs[i] for i in ids]

        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
        ])
        self.image_size = image_size

        self.model = torch.load('submodels/atten_0706.pth', map_location='cpu').eval()
        self.contra_model = torch.load('submodels/net_g_model_epoch_200.pth', map_location='cpu').eval()
        self.contra_pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        self.contra_projection = self.contra_model.projection

        self.kernal_size = 13
        self.conv = nn.Conv2d(1,1,self.kernal_size,1,self.kernal_size//2)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size, self.kernal_size))
        self.conv.bias = nn.Parameter(torch.Tensor([0]))

        self.kernal_size_2 = 13
        self.conv_2 = nn.Conv2d(1,1,self.kernal_size_2,1,self.kernal_size_2//2)
        self.conv_2.weight = nn.Parameter(torch.ones(1, 1, self.kernal_size_2, self.kernal_size_2))
        self.conv_2.bias = nn.Parameter(torch.Tensor([0]))

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        id = path.split("/")[-1]
        self.path = path
        slice = path.split('_')[-1].split('.')[0]
        c_feature = self._get_cls_free_feature(path.replace("/ap/", "/bp/"), slice).detach().cpu().squeeze().numpy()
        # c_feature = self._get_cls_free_feature(path, slice).detach().cpu().squeeze().numpy()

        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5

        if int(slice) - 1 >= 0:
            prev_img = tiff.imread(path.replace(f'_{slice}.', f'_{str(int(slice) - 1).zfill(3)}.'))
            prev_img = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
            prev_img = (prev_img - 0.5) / 0.5
        else:
            prev_img = np.zeros(img.shape)

        transformed = self.tfs(image=img)
        transformed_prev = self.tfs(image=prev_img)
        img = torch.unsqueeze(torch.Tensor(transformed["image"]), 0)
        prev_img = torch.unsqueeze(torch.Tensor(transformed_prev["image"]), 0)
        mask, one_hot_pred = self.get_mask_from_eff_mean(tiff.imread(os.path.join(self.eff_root, id)), # .replace('bp', 'apeff/apeff').replace('/ap/', '/apeff/')
                                         tiff.imread(os.path.join(self.mean_root, id)),
                                         img=img
                                        )
        mask = torch.unsqueeze(mask, 0)
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask
        # print(cond_image.shape)
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['prev_image'] = prev_img
        ret['mask'] = mask
        ret['one_hot_seg'] = one_hot_pred
        ret['slice'] = int(slice)
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['cemb'] = c_feature
        return ret

    def __len__(self):
        return len(self.imgs)

    def _get_cls_free_feature(self, path, ori_slice):
        subject_list = []
        for i in range(23):
            slice = str(i).zfill(3)
            subject_list.append(path.replace(f'_{ori_slice}.', f'_{slice}.'))
        subject = [tiff.imread(x) for x in subject_list]
        subject = np.stack(subject, 0)
        subject = subject / subject.max()
        subject = torch.from_numpy(subject).float()
        subject = subject.unsqueeze(1)
        subject = self.contra_model(subject, alpha=1, method='encode')[-1]
        subject = subject.permute(1, 2, 3, 0).unsqueeze(0)
        subject = self.contra_pool(subject)[:, :, 0, 0, 0]
        cfeature = self.contra_projection(subject)
        return cfeature

    def get_mask_from_eff_mean(self, mask, mask_2=None, img=None):
        threshold = 0
        # Blur effusion mask
        mask = self.conv(torch.Tensor(mask).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask = np.array(mask > threshold).astype(np.uint8)
        mask = torch.Tensor(mask)
        if img is not None:
            img = torch.unsqueeze(img, 0)
            pred = self.model(img)
            pred = torch.argmax(pred, 1, True)
            one_hot_pred = F.one_hot(pred, 3)
            one_hot_pred = torch.permute(one_hot_pred, (0, 4, 2, 3, 1)).squeeze(0).squeeze(-1).to(torch.float)
            # tiff.imwrite(self.path.replace("/ap/", "/apseg/").replace("/bp/", "/bpseg/"), one_hot_pred.detach().numpy().astype(np.uint8))

        if mask_2 is not None:
            # Blur lesion mask
            random_float = 0.06

            threshold = random_float * self.kernal_size_2 * self.kernal_size_2
            mask_2 = self.conv_2(torch.Tensor(mask_2).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            mask_2 = np.array(mask_2 > threshold).astype(np.uint8)
            mask_2 = torch.Tensor(mask_2)
            if 1:
                mask += mask_2
            if 0:
                mask = mask_2 * pred[0,0,::]
                # mask = mask_2 - mask

            mask = np.array(mask > 0).astype(np.uint8)
            mask = torch.Tensor(mask)

        if img is not None:
            return mask, one_hot_pred
        else:
            return mask


if __name__ == "__main__":
    train_dataset = PainDataset(["/media/ExtHDD01/Dataset/OAI_pain/full/full/bp/*", "/media/ExtHDD01/Dataset/OAI_pain/full/full/ap/*"], mask_config={"mask_mode": "hybrid"}, mode="evalll")

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=4,
                                       num_workers=10, drop_last=True)
    for i in train_dataloader:
        pass
        # print(i["gt_image"].shape, i["gt_image"].max(), i["gt_image"].min())
        # print(i["cond_image"].shape)
        # print(i["mask_image"].shape)
        # assert 0