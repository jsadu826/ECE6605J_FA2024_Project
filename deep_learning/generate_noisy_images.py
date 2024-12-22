import os
import cv2
import numpy as np
from PIL import Image

def add_gaussian_noise(im, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = im + (sigma * np.random.randn(*im.shape)).astype(np.int16)
    im = np.clip(im, 0., 255., out=None)
    im = im.astype(np.uint8)
    return im

for clean_img in os.listdir('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/clean'):
    clean_arr = cv2.imread(os.path.join('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/clean',clean_img),cv2.IMREAD_GRAYSCALE)
    for sigma in [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]:
        noisy_arr=add_gaussian_noise(clean_arr,sigma)
        save_dir = f'/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/sigma_{sigma}'
        os.makedirs(save_dir,exist_ok=True)
        Image.fromarray(noisy_arr,mode='L').save(os.path.join(save_dir,clean_img))
        
        