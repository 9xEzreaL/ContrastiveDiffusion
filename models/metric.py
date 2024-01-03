import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from core.Metric import Metric

import lpips
from torchvision.models.inception import inception_v3
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import numpy as np
from scipy.stats import entropy

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def ssim_score(inf_imgs, gen_imgs, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(inf_imgs)
    n = len(gen_imgs)

    assert batch_size > 0
    assert N > batch_size
    assert n == N
    # Set up dtype
    dtype = torch.FloatTensor

    # Set up dataloader
    inf_dataloader = torch.utils.data.DataLoader(inf_imgs, batch_size=batch_size)
    gen_dataloader = torch.utils.data.DataLoader(gen_imgs, batch_size=batch_size)

    ssim_score = []
    ms_ssim_score = []
    for i, (inf_img, gen_img) in enumerate(zip(inf_dataloader, gen_dataloader)):
        inf_img = inf_img.type(dtype)
        gen_img = gen_img.type(dtype)
        if inf_img.min() < 0:
            inf_img = (inf_img + 1) / 2
        if gen_img.min() < 0:
            gen_img = (gen_img + 1) / 2
        ssim_val = ssim(inf_img, gen_img, data_range=1, size_average=False, nonnegative_ssim=True)
        ms_ssim_val = ms_ssim(inf_img, gen_img, data_range=1, size_average=False)

        ssim_score.append(np.array(ssim_val))
        ms_ssim_score.append(np.array(ms_ssim_val))

    ssim_score = np.concatenate(ssim_score, 0)
    ms_ssim_score = np.concatenate(ms_ssim_score, 0)

    return np.mean(ssim_score), np.mean(ms_ssim_score)


def lpips_score(inf_imgs, gen_imgs, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(inf_imgs)
    n = len(gen_imgs)

    assert batch_size > 0
    assert N > batch_size
    assert n == N
    # Set up dtype
    dtype = torch.FloatTensor

    # Set up dataloader
    inf_dataloader = torch.utils.data.DataLoader(inf_imgs, batch_size=batch_size)
    gen_dataloader = torch.utils.data.DataLoader(gen_imgs, batch_size=batch_size)

    lpips_score = []
    for i, (inf_img, gen_img) in enumerate(zip(inf_dataloader, gen_dataloader)):
        inf_img = inf_img.type(dtype)
        gen_img = gen_img.type(dtype)

        loss_fn_lpips = lpips.LPIPS(net='alex')
        d = loss_fn_lpips(inf_img, gen_img)

        lpips_score.append(d.detach().numpy())

    lpips_score = np.concatenate(lpips_score, 0)

    return np.mean(lpips_score)

def dice_iou_cal_culator(inf_imgs, gen_imgs, model, batch_size=6):
    metric = Metric(num_classes=2)
    model.eval()
    N = len(inf_imgs)
    n = len(gen_imgs)

    assert batch_size > 0
    assert N > batch_size
    assert n == N

    # Set up dataloader
    inf_dataloader = torch.utils.data.DataLoader(inf_imgs, batch_size=batch_size)
    gen_dataloader = torch.utils.data.DataLoader(gen_imgs, batch_size=batch_size)

    for i, (inf_img, gen_img) in enumerate(zip(inf_dataloader, gen_dataloader)):
        inf_img = inf_img[:, 0:1, ::].cuda()
        gen_img = gen_img[:, 0:1, ::].cuda()

        inf_pred = model(inf_img)
        gen_img = model(gen_img)

        metric.update(inf_pred, gen_img)
    return metric.dice(), metric.iou()

