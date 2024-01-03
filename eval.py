import argparse
import torch
from cleanfid import fid
from core.base_dataset import BaseDataset, PainBaseDataset
from models.metric import inception_score, ssim_score, lpips_score, dice_iou_cal_culator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')
   
    ''' parser configs '''
    args = parser.parse_args()

    fid_score = fid.compute_fid(args.src, args.dst)
    print('FID: {}'.format(fid_score))

    is_mean, is_std = inception_score(PainBaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)
    print('IS:{} {}'.format(is_mean, is_std))

    ssim_score, ms_ssim_score = ssim_score(PainBaseDataset(args.src), PainBaseDataset(args.dst))
    print('SSIM score: {}, MS_SSIM score: {}'.format(ssim_score, ms_ssim_score))

    lpips_score = lpips_score(PainBaseDataset(args.src), PainBaseDataset(args.dst))
    print('LPIPS score: {}'.format(lpips_score))

    tse_bone_model = torch.load("submodels/atten_0706.pth")
    dice, iou = dice_iou_cal_culator(PainBaseDataset(args.src), PainBaseDataset(args.dst), tse_bone_model)
    print("dice: {}".format(dice))
    print("iou: {}".format(iou))

#