import os
import pickle

import cv2
import numpy as np
from scipy.ndimage import zoom

from bm3d_1st_step import bm3d_1st_step
from precompute_BM_zzy import precompute_BM_zzy
from psnr import compute_psnr
from utils import add_gaussian_noise, symetrize


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    relation_2d_indexed, relation_1d_indexed, similar_count_2d_indexed, similar_count_1d_indexed = precompute_BM_zzy(noisy_im, k_W, N_W, tauMatch_W)

    return img_basic, relation_2d_indexed, relation_1d_indexed, similar_count_2d_indexed, similar_count_1d_indexed


def process(gt_img_dir, gt_npy_dir, noisy_img_dir, noisy_npy_dir):
    os.makedirs(gt_img_dir, exist_ok=True)
    os.makedirs(gt_npy_dir, exist_ok=True)
    os.makedirs(noisy_img_dir, exist_ok=True)
    os.makedirs(noisy_npy_dir, exist_ok=True)

    cnt = 0
    for img_ent in os.scandir(gt_img_dir):
        cnt += 1

        img = cv2.imread(img_ent.path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype=np.uint8)
        if img.shape != (256, 256):
            img = zoom(img, (256 / img.shape[0], 256 / img.shape[1]))
        np.save(os.path.join(gt_npy_dir, img_ent.name.replace('.png', '.npy')), img)

        # for sigma in [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]:
        for sigma in [2]:
            print(f'------------ {cnt}th data, {img_ent.name}, sigma = {sigma} ------------')

            noisy_img = add_gaussian_noise(img, sigma)
            assert noisy_img.shape == (256, 256)
            cv2.imwrite(os.path.join(noisy_img_dir, img_ent.name.split('.')[0] + f'_sigma_{sigma}.png'), noisy_img)

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
            useSD_H = False  # FIXME wrong implementation?
            # -------------------------------------------------------------------------------
            # STEP 2 PARAMETERS -------------------------------------------------------------------------------
            tau_2D_W = 'DCT'
            k_W = 8
            N_W = 32
            p_W = 3 if sigma <= 40 else 6
            n_W = 19
            tauMatch_W = 400 if sigma <= 40 else 3500
            useSD_W = True  # FIXME
            # -------------------------------------------------------------------------------

            basic_img, relation_2d_indexed, relation_1d_indexed, similar_count_2d_indexed, similar_count_1d_indexed = run_bm3d(
                noisy_img, sigma, n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
            assert basic_img.shape == (256, 256)
            assert relation_2d_indexed.shape == (32, 32, N_W, 2), relation_2d_indexed.shape
            assert relation_2d_indexed.min() >= -1, relation_2d_indexed.min()
            assert relation_2d_indexed.max() < 32, relation_2d_indexed.max()
            assert relation_1d_indexed.shape == ((32 * 32), N_W), relation_1d_indexed.shape
            assert relation_1d_indexed.min() >= -1, relation_1d_indexed.min()
            assert relation_1d_indexed.max() < 1024, relation_1d_indexed.max()
            assert similar_count_2d_indexed.shape == (32, 32), similar_count_2d_indexed.shape
            assert similar_count_2d_indexed.min() == 0, similar_count_2d_indexed.min()
            assert similar_count_2d_indexed.max() <= 32, similar_count_2d_indexed.max()
            assert similar_count_1d_indexed.shape == (1024,), similar_count_1d_indexed.shape
            assert similar_count_1d_indexed.min() == 0, similar_count_1d_indexed.min()
            assert similar_count_1d_indexed.max() <= 32, similar_count_1d_indexed.max()
            data = {
                'noisy_img': noisy_img,
                'basic_img': basic_img,
                'relation_2d_indexed': relation_2d_indexed,
                'relation_1d_indexed': relation_1d_indexed,
                'similar_count_2d_indexed': similar_count_2d_indexed,
                'similar_count_1d_indexed': similar_count_1d_indexed
            }
            with open(os.path.join(noisy_npy_dir, img_ent.name.split('.')[0] + f'_sigma_{sigma}.npy'), 'wb') as f:
                pickle.dump(data, f)


if __name__ == '__main__':
    gt_img_dir = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/clean'
    gt_npy_dir = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/test_bm3d_images/gt_npy'
    noisy_img_dir = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/test_bm3d_images/noisy_png'
    noisy_npy_dir = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/test_bm3d_images/noisy_npy'
    process(gt_img_dir, gt_npy_dir, noisy_img_dir, noisy_npy_dir)
