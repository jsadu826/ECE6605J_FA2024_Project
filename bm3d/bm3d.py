from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    # no need the following two lines, already defined in __main__
    # k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    # k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np

    im_dir = 'test_data/image'
    save_dir = 'tmp'
    os.makedirs(save_dir, exist_ok=True)
    # for im_name in os.listdir(im_dir):
    for im_name in ['Cameraman.png',]:
        # sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]
        sigma_list = [40]
        for sigma in sigma_list:
            print(im_name, '  ', sigma)

            # STEP 1 PARAMETERS -------------------------------------------------------------------------------
            tau_2D_H = 'BIOR' if sigma <= 40 else 'DCT'  # TAU_2D: 2D transform method
            k_H = 8 if sigma <= 40 else 12  # N_1: patch size
            N_H = 16  # N_2: number of patches to be stacked into a 3D group
            p_H = 3 if sigma <= 40 else 4  # N_step: the reference patches are selected every p_H steps (pixels) in both x, y directions
            n_H = 19  # N_S = 2 * n_H + 1: search similar patches in N_S * N_S area
            # N_FS, N_PR, beta: hard coded, just ignore
            # lambda_2D: threshold for 2D hard thresholding # FIXME - missing in code
            lambda3D_H = 2.7 if sigma <= 40 else 2.8  # lambda_3D: threshold for 3D hard thresholding
            tauMatch_H = 2500 if sigma <= 40 else 5000  # tau_match: threshold that whether determinates two patches are similar
            useSD_H = True  # FIXME wrong implementation?
            # -------------------------------------------------------------------------------

            # STEP 2 PARAMETERS -------------------------------------------------------------------------------
            tau_2D_W = 'DCT'
            k_W = 8 if sigma <= 40 else 11
            N_W = 32
            p_W = 3 if sigma <= 40 else 6
            n_W = 19
            tauMatch_W = 400 if sigma <= 40 else 3500
            useSD_W = True  # FIXME
            # -------------------------------------------------------------------------------

            noisy_dir = 'test_data/sigma' + str(sigma)

            im_path = os.path.join(im_dir, im_name)
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            noisy_im_path = os.path.join(noisy_dir, im_name)
            noisy_im = cv2.imread(noisy_im_path, cv2.IMREAD_GRAYSCALE)

            im1, im2 = run_bm3d(noisy_im, sigma,
                                n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                                n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

            psnr_1st = compute_psnr(im, im1)
            psnr_2nd = compute_psnr(im, im2)

            im1 = (np.clip(im1, 0, 255)).astype(np.uint8)
            im2 = (np.clip(im2, 0, 255)).astype(np.uint8)

            save_name = im_name[:-4] + '_s' + str(sigma) + '_py_1st_P' + '%.4f' % psnr_1st + '.png'
            cv2.imwrite(os.path.join(save_dir, save_name), im1)
            save_name = im_name[:-4] + '_s' + str(sigma) + '_py_2nd_P' + '%.4f' % psnr_2nd + '.png'
            cv2.imwrite(os.path.join(save_dir, save_name), im2)
