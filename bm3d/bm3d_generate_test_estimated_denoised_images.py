import os
import pickle

import cv2
import numpy as np

from bm3d_1st_step import bm3d_1st_step
from PIL import Image
from utils import symetrize


def run_bm3d(noisy_im, sigma, n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H):
    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]
    return img_basic


if __name__ == '__main__':
    
    for sigma in [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]:
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
        # -------------------------------------------------------------------------------------------------
        save_dir = f'/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/basic_sigma_{sigma}'
        os.makedirs(save_dir,exist_ok=True)
        for img_ent in os.scandir(f'/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/sigma_{sigma}'):
            img = cv2.imread(img_ent.path, cv2.IMREAD_GRAYSCALE)
            img = np.array(img, dtype=np.uint8)
            basic_img=run_bm3d(img, sigma, n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H).astype(np.uint8)
            print(basic_img.shape)
            Image.fromarray(basic_img,mode='L').save(os.path.join(save_dir,img_ent.name))