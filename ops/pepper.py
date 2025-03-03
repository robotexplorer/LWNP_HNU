#自定义transform
#加入椒盐噪声
import random
from PIL import Image
import numpy as np
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            c, h, w = img_.shape
            # print(c)
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=0)
            img_[mask == 1] = 1  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return img_
            # return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img
#添加高斯噪声
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img

'''
# img=Image.open("balloons_RGB.bmp")
import scipy.io as sio
img = sio.loadmat('0_0.mat')['img_hs']
import torch
img = torch.from_numpy(img)
img = img.permute(2,0,1)
img = img.numpy()

# img1=AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45))(img)
# img1.show()
img2=AddPepperNoise(0.99,0.9)(img)
# print(img2.shape)
import cv2
cv2.imshow('00.bmp',img2[15,:,:])
cv2.waitKey(1000)
# img2.show()
'''