import tifffile as tiff
import numpy as np
import glob
import os

def to8bit(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img
key = 'all'
path = f'/media/ExtHDD01/logs/0901-cosine-500step-pain-cond/superpain/all/results/0/{key}/*'
os.makedirs(path.replace(key, f'{key}_8bit').replace('*', ''), exist_ok=True)

for i in glob.glob(path):
    img = tiff.imread(i)
    img = to8bit(img)
    tiff.imwrite(i.replace(key, f'{key}_8bit'), img)