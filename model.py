import torch
from func import *
device = torch.device('cuda')
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'

from model_loader import init_model, load_model
import torch.nn as nn
import torch
import cv2
import matplotlib.pyplot as plt
from thop import profile
from ops.utils_blocks import block_module
import time

model1 = init_model(in_channels=1,channels=16,num_half_layer=5,rs=0).to(device)
model1 = nn.DataParallel(model1)

dirs1 = 'trained_model/'
ckpts = 'ckpt_30'
model_ckpt = dirs1+ckpts


####                                 GAP                                  ####
def gap_denoise(meas, Phi, data_truth, ref_img, matfile, args):

    pass


####                                ADMM                                  ####
def admm_denoise(meas, Phi, data_truth, ref_img, matfile, args):
    #-------------- Initialization --------------#
    if args.x0 is None:
        x0 = At(meas, Phi)
    iter_max = [args.iter_max] * len(args.sigma)
    ssim_all = []
    psnr_all = []
    k = 0
    show_iqa = True
    noise_estimate = True
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1

    x = x0.to(device)
    theta = x0.to(device)
    b = torch.zeros_like(x0).to(device)
    gamma = 0.01

    # ---------------- Iteration ----------------#
    for idx, noise_level in enumerate(args.sigma):
        for iter in range(iter_max[idx]):
            # Euclidean Projection
            theta = theta.to(device)
            b = b.to(device)
            meas_b = A(theta+b, Phi)
            x = (theta + b) + args.lambda_*(At((meas - meas_b)/(Phi_sum + gamma), Phi))
            x1 = shift_back(x-b, step=2)

            if args.denoiser == 'TV-LWNP':
                if k > 80 and k < 1500:#k 
                    load_model(model_name=model_ckpt, model=model1, device_name=device_name)
                    with torch.no_grad():
                        model1.eval()
                        x1 = torch.unsqueeze(x1, 0).permute(0, 3, 1, 2)
                        if args.block_inference==False:
                            x1 = model1(x1)
                        else:
                            params={
                            'crop_out_blocks': 0,
                            'ponderate_out_blocks': 1,
                            'sum_blocks': 0,
                            'pad_even': 1,  # otherwise pad with 0 for las
                            'centered_pad': 0,  # corner pixel have only one estimate
                            'pad_block': 1,  # pad so each pixel has S**2 estimate #1
                            'pad_patch': 0,  # pad so each pixel from the image has at least S**2 estimate from 1 block #0
                            'no_pad': False,
                            'custom_pad': None,
                            'avg': 1}
                            block = block_module(args.patch_size, args.stride_test, args.kernel_size, params)
                            noisy_blocks = block._make_blocks(x1)#torch.Size [625,28,32,32]
                            patch_loader = torch.utils.data.DataLoader(noisy_blocks, batch_size=1, drop_last=False)
                            out_blocks = torch.zeros_like(noisy_blocks)
                            for i, inp in enumerate(patch_loader):
                                id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
                                out_blocks[id_from:id_to] = model1(inp)#[1,28,32,32]
                            x1 = block._agregate_blocks(out_blocks)
                        theta = torch.squeeze(x1, 0).permute(1, 2, 0)
                        theta_s = theta.clone()
                else:
                    # theta = TV_minimization(x1, ref_img, args.tv_weight, args.tv_iter_max)
                    theta = TV_denoiser(x1, args.tv_weight, args.tv_iter_max)
                    #x1 = x1 - ref_img
                #theta = TV_denoiser(x1, args.tv_weight, args.tv_iter_max)
                #theta = theta + ref_img
            elif args.denoiser == 'TV':
                theta = TV_denoiser(x1, args.tv_weight, args.tv_iter_max)    
            # --------------- Evaluation ---------------#
            if show_iqa and data_truth is not None:
                ssim_all.append(calculate_ssim(data_truth, theta))
                psnr_all.append(calculate_psnr(data_truth, theta))
                if (k+1)>70 and (k + 1) % 5 == 0:
                    print('  ADMM-{0} iteration {1: 3d}, '
                          'PSNR {2:2.2f} dB.'.format(args.denoiser.upper(), k + 1, psnr_all[k]),
                          'SSIM:{0:0.4f}'.format(ssim_all[k]))
            theta1 = theta
            theta = shift(theta, step=2)
            b = b - (x.to(device) - theta.to(device))
            k += 1
    return theta1, psnr_all, ssim_all, dirs1+'_'+ckpts
