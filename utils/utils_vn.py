

import torch
import torch.nn as nn
from torch.autograd import Function as autoF
from scipy.special import gammaln
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import img_as_ubyte
import numpy as np
import sys
from math import floor
import math
import cv2
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale

import random
import math
from torch.autograd import Variable
import torch

import torchvision.transforms as transforms

# gray = transforms.Gray()
import numpy as np


def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''
    if im1.ndim == 2:
        out = compare_ssim(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
    elif im1.ndim == 3:
        out = compare_ssim(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=255)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += ssim_index(Iclean[i,:,:,:].transpose((1,2,0)), Img[i,:,:,:].transpose((1,2,0)))
    return (SSIM/Img.shape[0])

def peaks(n):
    '''
    Implementation the peak function of matlab.
    '''
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    [XX, YY] = np.meshgrid(X, Y)
    ZZ = 3 * (1-XX)**2 * np.exp(-XX**2 - (YY+1)**2) \
            - 10 * (XX/5.0 - XX**3 -YY**5) * np.exp(-XX**2-YY**2) - 1/3.0 * np.exp(-(XX+1)**2 - YY**2)
    return ZZ

def generate_gauss_kernel_mix(H, W):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    pch_size = 32
    K_H = floor(H / pch_size)
    K_W = floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))
    centerW = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)
    centerH = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)
    scale = np.random.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    scale = scale.astype(np.float32)
    XX, YY = np.meshgrid(np.arange(0, W), np.arange(0,H))
    XX = XX[:, :, np.newaxis].astype(np.float32)
    YY = YY[:, :, np.newaxis].astype(np.float32)
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerW)**2-(YY-centerH)**2)/(2*scale**2) )
    out = ZZ.sum(axis=2, keepdims=False) / K

    return out

def sincos_kernel():
    # Nips Version
    [xx, yy] = np.meshgrid(np.linspace(1, 10, 512), np.linspace(1, 20, 512))####default=256
    # [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(-10, 15, 256))
    zz = np.sin(xx) + np.cos(yy)
    return zz

def capacity_cal(net):
    out = 0
    for param in net.parameters():
        out += param.numel()*4/1024/1024
    # print('Networks Parameters: {:.2f}M'.format(out))
    return out

class LogGamma(autoF):
    '''
    Implement of the logarithm of gamma Function.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output

        return grad_input

def load_state_dict_cpu(net, state_dict0):
    state_dict1 = net.state_dict()
    for name, value in state_dict1.items():
        assert 'module.'+name in state_dict0
        state_dict1[name] = state_dict0['module.'+name]
    net.load_state_dict(state_dict1)



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1_temp = im1.astype(np.float64)
    im2_temp = im2.astype(np.float64)
    mse = np.mean((im1_temp - im2_temp)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_valid_crop_size(crop_size, blocksize):
    return crop_size - (crop_size % blocksize)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        Grayscale(),
        ToTensor(),
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, phi,crop_size, blocksize):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, blocksize)
        self.hr_transform = train_hr_transform(crop_size)
        self.phi=phi

    def __getitem__(self, index):
        try:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
            H = hr_image.shape[1]  #######hang
            W = hr_image.shape[2]
            img_x1 = hr_image.reshape(1, H * W)
            y = torch.mm(self.phi, img_x1.T)  # Performs a matrix-vector product
            return y, hr_image

        except:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index+1]))

            H = hr_image.shape[1]  #######hang
            W = hr_image.shape[2]
            img_x1 = hr_image.reshape(1, H * W)
            y = torch.mm(self.phi, img_x1.T)  # Performs a matrix-vector product
            return y, hr_image
    def __len__(self):
        return len(self.image_filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir,phi, blocksize):
        super(TestDatasetFromFolder, self).__init__()
        self.blocksize = blocksize
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

        self.phi = phi
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])

        w, h = hr_image.size
        w = int(np.floor(w/self.blocksize)*self.blocksize)
        h = int(np.floor(h/self.blocksize)*self.blocksize)
        crop_size = (32, 32)

        hr_image = CenterCrop(crop_size)(hr_image)
        hr_image = Grayscale()(hr_image)
        hr_image=ToTensor()(hr_image)
        H = hr_image.shape[1]  #######hang
        W = hr_image.shape[2]
        img_x1 = hr_image.reshape(1, H * W)
        y = torch.mm(self.phi, img_x1.T)  # Performs a matrix-vector product
        return y, hr_image

        # return ToTensor()(hr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

