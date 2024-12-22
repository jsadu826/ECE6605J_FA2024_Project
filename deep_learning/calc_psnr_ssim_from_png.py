import cv2
import numpy as np
import torch

from restormer_orig.utils import psnr, ssim


def add_gaussian_noise(im, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = im + (sigma * np.random.randn(*im.shape)).astype(np.int16)
    im = np.clip(im, 0., 255., out=None)
    im = im.astype(np.uint8)
    return im


with torch.no_grad():
    img_names = ['Cameraman.png', 'house.png', 'Peppers.png', 'montage.png', 'Lena.png', 'barbara.png', 'boat.png', 'fingerprint.png', 'Man.png', 'couple.png', 'hill.png']
    sigmas = [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]
    all_psnr = np.zeros((11, 11), dtype=np.float32)
    all_ssim = np.zeros((11, 11), dtype=np.float32)

    for i, sigma in enumerate(sigmas):
        print(sigma, end=' ')
        for j, img_name in enumerate(img_names):
            clean_orig = np.array(cv2.imread('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/clean/' + img_name, cv2.IMREAD_GRAYSCALE))
            denoised_orig = np.array(cv2.imread(f'/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_repro_images/sigma_{sigma}/' + img_name, cv2.IMREAD_GRAYSCALE))

            clean_tensor = torch.from_numpy(clean_orig).unsqueeze(0).unsqueeze(0).to(torch.float32).cuda()
            denoised_tensor = torch.from_numpy(denoised_orig).unsqueeze(0).unsqueeze(0).to(torch.float32).cuda()

            current_psnr, current_ssim = psnr(denoised_tensor, clean_tensor), ssim(denoised_tensor, clean_tensor)
            current_psnr, current_ssim = round(float(current_psnr), 2), round(float(current_ssim * 100), 2)

            all_psnr[i][j] = current_psnr
            all_ssim[i][j] = current_ssim

            print('& \\textcolor{{blue}}{{{}}}/\\textcolor{{red}}{{{}}}'.format(current_psnr, current_ssim), end=' ')
        print('\\\\ \\hline')

    np.save('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_repro_images_psnr.npy', all_psnr)
    np.save('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_repro_images_ssim.npy', all_ssim)
