import cv2
import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TVTF
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, data, phi,length):
        self.data = data
        self.len = length
        self.phi=phi
    def __getitem__(self, index):
        img=torch.Tensor(self.data[index, :]).float()
        img_x1 = img.reshape(1, 32 * 32)
        y = torch.mm(self.phi, img_x1.T)  # Performs a matrix-vector product
        img = img.unsqueeze(dim=0)

        return y,img

    def __len__(self):
        return self.len


def npimg_to_tensor(img):
    npimg_to_tensor_transforms = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0], HWC->CHW
    ])
    return npimg_to_tensor_transforms(img.copy())


def pad_image(img, d=8):
    """
    padding imgs such that it can be input of Net
    """
    C, H, W = img.size()
    if H % 8 == 0:
        pad_H = 0
    else:
        pad_H = (H // d + 1) * d - H
    if W % 8 == 0:
        pad_W = 0
    else:
        pad_W = (W // d + 1) * d - W
    pad = (0, pad_W, 0, pad_H)
    img = F.pad(img, pad, 'constant', 0)
    return img


def pad_kernel(kernel, max_size=31):
    """
    padding kernel to same size
    kernel: pytorch tensor
    """
    C, H, W = kernel.size()
    pad_L = (max_size - W) // 2
    pad_R = max_size - pad_L - W
    pad_U = (max_size - H) // 2
    pad_D = max_size - pad_U - H
    pad = (pad_L, pad_R, pad_U, pad_D)
    img = F.pad(kernel, pad, 'constant', 0)
    return img


def rgb_to_yuv(x):
    '''convert batched rgb tensor to yuv'''
    out = torch.zeros_like(x)
    out[:,0,:,:] =  0.299    * x[:,0,:,:] + 0.587    * x[:,1,:,:] + 0.114   * x[:,2,:,:]
    out[:,1,:,:] = -0.168736 * x[:,0,:,:] - 0.331264 * x[:,1,:,:] + 0.5   * x[:,2,:,:]
    out[:,2,:,:] =  0.5      * x[:,0,:,:] - 0.418688 * x[:,1,:,:] - 0.081312 * x[:,2,:,:]
    return out


def npimg_data_augmentation(image, mode):
    if mode == 0:
        pass
    elif mode == 1:
        out = np.rot90(image)
    elif mode == 2:
        out = np.rot90(image, k=2)
    elif mode == 3:
        out = np.rot90(image, k=3)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


def npimg_random_augmentation(*args):
    out = []
    if random.randint(0,1) == 1:
        flag_aug = random.randint(1, 3)
        for data in args:
            out.append(npimg_data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data.copy())
    return out


def tensorimg_random_augmentation(*args):
    out = []
    aug = random.randint(0, 3)
    if aug == 0:
        for data in args:
            out.append(torch.rot90(data, dims=(1,2)))
    elif aug == 1:
        for data in args:
            out.append(torch.rot90(data, dims=(1, 2), k = 2))
    elif aug == 2:
        for data in args:
            out.append(torch.rot90(data, dims=(1, 2), k = 3))
    else:
        for data in args:
            out.append(data)
    return out


def tensorimg_add_noise(img, sigma=1.0, rgb_range=255.0):
    # img: pytorch tensor
    noise = torch.randn_like(img) * sigma / rgb_range
    return (img + noise).clamp(0.0, 1.0)


def npimg_random_crop_patch(im, patch_size):
    H = im.shape[0]
    W = im.shape[1]
    if H < patch_size or W < patch_size:
        H = max(patch_size, H)
        W = max(patch_size, W)
        im = cv2.resize(im, (W, H))
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch = im[ind_H : ind_H + patch_size, ind_W : ind_W + patch_size]
    return patch


def tensorimg_random_crop_patch(im, patch_size):
    H = im.shape[0]
    W = im.shape[1]
    if H < patch_size or W < patch_size:
        H = max(patch_size, H)
        W = max(patch_size, W)
        im = cv2.resize(im, [H, W])
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch = im[ ind_H : ind_H + patch_size, ind_W : ind_W + patch_size]

    return patch


def npimg_random_crop_pair_patch(im1, im2, patch_size):
    H = im1.shape[0]
    W = im2.shape[1]
    if H < patch_size or W < patch_size:
        H = max(patch_size, H)
        W = max(patch_size, W)
        im1 = cv2.resize(im1, (W, H))
        im2 = cv2.resize(im2, (W, H))
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch1 = im1[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
    patch2 = im2[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
    return patch1, patch2


def tensorimg_random_crop_pair_patch(im1, im2, patch_size):
    """
    im1, im2: tensor
    return: tensor
    """
    H = im1.shape[0]
    W = im1.shape[1]
    if H < patch_size or W < patch_size:
        H = max(patch_size, H)
        W = max(patch_size, W)
        im1 = TVTF.resize(im1, [H, W])
        im2 = TVTF.resize(im2, [H, W])
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch1 = im1[:, ind_H : ind_H + patch_size, ind_W : ind_W + patch_size]
    patch2 = im2[:, ind_H : ind_H + patch_size, ind_W : ind_W + patch_size]
    return patch1, patch2



def tensor_img_to_patch(img, patch_h = 300, patch_w = 300, step_h = 70, step_w = 70):
    assert len(img.size()) == 4 and img.size()[0] == 1, "Input shape must be 1CHW"
    # img 1 x C x H x W : --> N x C x patch_H x patch_W
    B, C, H, W = img.shape
    assert ((H - patch_h) % step_h) == 0, "Invalid patch_h and step_h"
    assert ((W - patch_w) % step_w) == 0, "Invalid patch_h and step_h"
    n_H_patch = (H - patch_h) // step_h + 1
    n_W_patch = (W - patch_w) // step_w + 1
    out = torch.zeros(n_W_patch * n_H_patch, C, patch_h, patch_w).to(img.device)
    mask = torch.zeros_like(img)
    num = 0
    for H_i in range(n_H_patch):
        for W_i in range(n_W_patch):
            ind_H = H_i * step_h
            ind_W = W_i * step_w
            patch = img[:, :, ind_H: ind_H + patch_h, ind_W: ind_W + patch_w]
            out[num] = patch.squeeze(0)
            num += 1
            mask[:, :, ind_H: ind_H + patch_h, ind_W: ind_W + patch_w] = mask[:, :, ind_H: ind_H + patch_h, ind_W: ind_W + patch_w] + 1
    return out, mask


def tensor_patch_to_img(patch, mask, step_h = 70, step_w = 70):
    N, C, patch_h, patch_w = patch.shape
    img = torch.zeros_like(mask)
    _, _, img_h, img_w = mask.shape
    n_H_patch = (img_h - patch_h) // step_h + 1
    n_W_patch = (img_w - patch_w) // step_w + 1
    index = 0
    for H_i in range(n_H_patch):
        for W_i in range(n_W_patch):
            ind_H = H_i * step_h
            ind_W = W_i * step_w
            img[:, :, ind_H: ind_H + patch_h, ind_W: ind_W + patch_w] = img[:, :, ind_H: ind_H + patch_h, ind_W: ind_W + patch_w] + patch[index]
            index += 1
    return img / mask



def gopro_to_pair_patch(img1, img2):
    # img 1280 x 720 : B x C x H x W --> NB x C x patch_H x patch_W
    # make sur B == 1 and im1.shape == img2.shape
    B, C, H, W = img1.shape
    patch_H = 360
    patch_W = 320
    n_H_patch = 4
    n_W_patch = 8
    out1 = torch.zeros(n_W_patch * n_H_patch, C, patch_H, patch_W).to(img1.device)
    out2 = torch.zeros(n_W_patch * n_H_patch, C, patch_H, patch_W).to(img1.device)
    num = 0
    for H_i in range(n_H_patch):
        for W_i in range(n_W_patch):
            ind_H = H_i * patch_H
            ind_W = W_i * patch_W
            patch1 = img1[:, :, ind_H: ind_H + patch_H, ind_W: ind_W + patch_W]
            out1[num] = patch1.squeeze(0)
            patch2 = img2[:, :, ind_H: ind_H + patch_H, ind_W: ind_W + patch_W]
            out2[num] = patch2.squeeze(0)
    return out1, out2



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count








