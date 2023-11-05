import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import glob
import albumentations as A
import tifffile as tiff

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

class BaseDataset(data.Dataset):
    def __init__(self, data_root, image_size=[256, 256], loader=pil_loader):
        self.imgs = make_dataset(data_root)
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        return img

    def __len__(self):
        return len(self.imgs)


class PainBaseDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[384, 384], loader=pil_loader):
        if not data_root.__contains__('*'):
            data_root = os.path.join(data_root, "*")
        imgs = sorted(glob.glob(data_root))

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = A.Compose([
            A.Resize(width=image_size[0], height=image_size[1]),
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = tiff.imread(path)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0.5) / 0.5
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
            img = np.concatenate([img, img, img], 0)

        return img

    def __len__(self):
        return len(self.imgs)