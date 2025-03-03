import os
import time
import torch
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
from model import gap_denoise, admm_denoise
from torchmetrics import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis
import cv2

device = torch.device('cuda')
random.seed(5)

#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ADMM', help="Select GAP or ADMM")
parser.add_argument('--lambda_', default=1, help="_lambda is the regularization factor")
parser.add_argument('--denoiser', default='TV-LWNP', help="Select which denoiser: Total Variation (TV) or (TV-LWNP)")
parser.add_argument('--accelerate', default=True, help="Acclearted version of GAP or not")
parser.add_argument('--iter_max', default=300, help="Maximum number of iterations")
parser.add_argument('--tv_weight', default=3, help="TV denoising weight (larger for smoother but slower)")
parser.add_argument('--tv_iter_max', default=10, help="TV denoising maximum number of iterations each")
parser.add_argument('--x0', default=None, help="The initialization data point")
parser.add_argument('--sigma', default=[55], help="The noise levels")
parser.add_argument('--block_inference', default=False, help="是否采取patch化分块去噪的方式")
parser.add_argument('--patch_size', default=64, help="patch化去噪时采用的patch尺寸")#
parser.add_argument('--stride_test', default=48, help="patch化去噪时采用的采样步长")#
parser.add_argument('--kernel_size', default=30, help="patch化去噪时采用的卷积核尺寸")#
args = parser.parse_args()
#------------------------------------------------------------------#
dirs = ''
data_list=os.listdir(dirs)
length=len(data_list)
dirs_refimg = ''

for tis in range(1):
    for ij in range(length):
        #----------------------- Data Configuration -----------------------#
        matfile = data_list[ij][:-4]
        h, w, nC, step = 256, 256, 28, 2
        data_truth = torch.from_numpy(sio.loadmat(dirs+matfile+'.mat')['img_hs'])
        data_truth = data_truth/torch.max(data_truth)
        data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC))
        for i in range(nC):
            data_truth_shift[:, i*step:i*step+256, i] = data_truth[:, :, i]

        ref_img = cv2.imread(dirs_refimg)
        ref_img = torch.from_numpy(ref_img).to(device)
        ref_img_1 = torch.unsqueeze(ref_img[:, :, 0], 2).repeat(1, 1, 9)
        ref_img_2 = torch.unsqueeze(ref_img[:, :, 1], 2).repeat(1, 1, 9)
        ref_img_3 = torch.unsqueeze(ref_img[:, :, 2], 2).repeat(1, 1, 10)
        ref_img = torch.cat((ref_img_1, ref_img_2, ref_img_3), dim=2)
        ref_img = ref_img/torch.max(ref_img)
        #------------------------------------------------------------------#


        #----------------------- Mask Configuration -----------------------#
        mask = torch.zeros((h, w + step*(nC - 1)))
        mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
        mask_256 = torch.from_numpy(sio.loadmat('mask256.mat')['mask'])
        for i in range(nC):
            mask_3d[:, i*step:i*step+256, i] = mask_256
        Phi = mask_3d
        meas = torch.sum(Phi * data_truth_shift, 2)

        begin_time = time.time()
        if args.method == 'GAP':
            pass
                
        elif args.method == 'ADMM':
            recon, psnr_all, ssim_all, modelParams_ckpts = admm_denoise(meas.to(device), Phi.to(device), data_truth.to(device), ref_img.to(device), matfile, args)
            end_time = time.time()
            
            psnr_all_np = np.zeros([args.iter_max,length])
            ssim_all_np = np.zeros([args.iter_max,length])
            for n in range(len(args.sigma)*args.iter_max):
                psnr_all_np[n,ij] = psnr_all[n].cpu().numpy()
                ssim_all_np[n,ij] = ssim_all[n].cpu().numpy()
            _psnr = calculate_psnr(data_truth, recon.cpu())
            _ssim = calculate_ssim(data_truth, recon.cpu())
            
            print('KAIST-{}-{} ADMM-{} PSNR {:2.4f} dB, SSIM {:2.4f}, running time {:.1f} seconds.'.format(
                tis, ij, args.denoiser.upper(), _psnr, _ssim, end_time - begin_time))
            results_dir = 'results/'+modelParams_ckpts+'/'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
        sio.savemat(results_dir+'Recon_'+matfile+'.mat', {'img_hs':recon.cpu().numpy()})