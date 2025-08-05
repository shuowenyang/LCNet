import torch
from math import log
from utils.utils_vn import LogGamma

log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)

def loss_fn(out_Net, im_gt,beta):


    B = im_gt.shape[0]
    C = im_gt.shape[1]



    mu = out_Net[:, :C, ]

    sigma2 = torch.exp(out_XNet[:, C:, ].clamp(min=log_min, max=log_max))

    sigma2_div_alpha2 = torch.div(sigma2, beta)
    sigma2_div_alpha2=sigma2_div_alpha2.view(B,C,96,96)


    kl_gauss = 0.5 * torch.mean((mu - im_gt) ** 2 / beta + (sigma2_div_alpha2 - 1 - torch.log(sigma2_div_alpha2)))





    return  kl_gauss

