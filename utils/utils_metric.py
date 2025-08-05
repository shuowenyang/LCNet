from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage import img_as_ubyte
import sys



def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=255)
    return (PSNR / Img.shape[0])


def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''
    if im1.ndim == 2:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
    elif im1.ndim == 3:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out


def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += ssim_index(Iclean[i,:,:,:].transpose((1,2,0)), Img[i,:,:,:].transpose((1,2,0)))
    return (SSIM/Img.shape[0])





# from utils_imgs import npimg_to_tensor
# im1 = cv2.imread("../data/lai/ground_truth/people_01.png")
# im2 = cv2.imread("../data/lai/uniform/people_01_kernel_01.png")
#
# s1 = batch_SSIM(npimg_to_tensor(im1).unsqueeze(0), npimg_to_tensor(im2).unsqueeze(0))
# s2 = ssim(npimg_to_tensor(im1).unsqueeze(0), npimg_to_tensor(im2).unsqueeze(0))
# print(s1)
# print(s2)
























