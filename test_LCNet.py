import numpy as np
import os
import glob
from time import time
import cv2
from skimage.metrics import structural_similarity as ssim
import argparse
from LCNet_model import *
import warnings
from ptflops import get_model_complexity_info
warnings.filterwarnings("ignore")
from thop import profile
from thop import clever_format
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import scipy.io as scio
import hdf5storage



def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LCNet(sensing_rate=args.sensing_rate, LayerNo=args.layer_num)
    model = nn.DataParallel(model)
    model = model.to(device)
    #
    # input=torch.randn(1,1,33,33).to(device)
    # flops, params = profile(model, inputs=(input,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))




    model_dir = "./%s/%s_group_%d_ratio_%.2f_0.0000_new_s5_a2_gamma0" % (args.save_dir, args.model, args.group_num, args.sensing_rate)
    checkpoint = torch.load("%s/net_params_%d.pth" % (model_dir, args.epochs), map_location=device)
    model.load_state_dict(checkpoint['net'])



    ext = {'/*.jpg', '/*.png', '/*.tif'}
    filepaths = []
    test_dir = os.path.join('./DataSets/', args.test_name)
    for img_type in ext:
        filepaths = filepaths + glob.glob(test_dir + img_type)

    result_dir = os.path.join(args.result_dir, args.test_name,'%.2f','s5_a2_new_gamma0')% ( args.sensing_rate)######################modify

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    Time_All = np.zeros([1, ImgNum], dtype=np.float32)

    with torch.no_grad():
        model(torch.zeros(1, 1, 256, 256).cuda())
        print("\nCS Reconstruction Start")
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]

            Img = cv2.imread(imgName, 1)
            # ########noisy
            # noise=np.random.normal(0,args.sigma,size=Img.size).reshape(Img.shape[0],Img.shape[1],Img.shape[2])
            # Img_noise=Img+noise
            # Img_noise=np.uint8(Img_noise)
            ##########

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)#############Img/Img_noise
            Img_rec_yuv = Img_yuv.copy()


            Iorg_y = Img_yuv[:, :, 0]
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Img_output = Ipad / 255.

            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_x = batch_x.unsqueeze(0).unsqueeze(0)

            start = time()
            output, sys_cons ,initial_y, initial_Phi = model(batch_x)
            end = time()

            print(sys_cons)

            x_output = output[:,:1,:].squeeze(0).squeeze(0)
            sigma_output=output[:,1:,:].squeeze(0).squeeze(0)



            Prediction_value = x_output.cpu().data.numpy()
            X_rec = np.clip(Prediction_value[:row, :col], 0, 1)

            Sigma_value = sigma_output.cpu().data.numpy()
            Sigma_value=normalize(Sigma_value,max_val=np.max(Sigma_value),min_val=np.min(Sigma_value))
            X_sigma = Sigma_value


            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)

            test_name_split = os.path.split(imgName)
            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
                img_no, ImgNum, test_name_split[1], (end - start), rec_PSNR, rec_SSIM))

            Img_rec_yuv[:, :, 0] = X_rec * 255
            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)



            im_sigma_rgb = np.clip(X_sigma* 255, 0, 255).astype(np.uint8)
            sigmaName = "./%s/%s" % (result_dir, test_name_split[1])
            cv2.imwrite("%s_CSratio_%.2f_epoch_%d.png" % (
                sigmaName, args.sensing_rate, args.epochs), im_sigma_rgb)

            resultName = "./%s/%s" % (result_dir, test_name_split[1])
            cv2.imwrite("%s_CSratio_%.2f_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (
                resultName, args.sensing_rate, args.epochs, rec_PSNR, rec_SSIM), im_rec_rgb)


            del x_output

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            Time_All[0, img_no] = end - start

    print('\n')
    output_data = "CS ratio is %.2f, Avg PSNR/SSIM/Time for %s is %.2f/%.4f/%.4f, Epoch number of model is %d \n" % (
        args.sensing_rate, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(Time_All), args.epochs)
    print(output_data)

    print("CS Reconstruction End")


def imread_CS_py(Iorg):
    block_size = args.block_size
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.math.log10(PIXEL_MAX / np.math.sqrt(mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LCNet', help='model name')
    parser.add_argument('--sensing-rate', type=float, default=0.1000, help='set sensing rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--block_size', type=int, default=32, help='block size (default: 32)')
    parser.add_argument('--save_dir', type=str, default='save_temp', help='The directory used to save models')
    parser.add_argument('--group_num', type=int, default=1, help='group number for training')
    parser.add_argument('--layer_num', type=int, default=16, help='D2fm number of the Net')
    parser.add_argument('--sigma', type=int, default=3, help='noise variance')
    parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
    parser.add_argument('--result_dir', type=str, default='result', help='result directory')
    main()
