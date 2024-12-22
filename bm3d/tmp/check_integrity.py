import os
import shutil

# for gt_npy_name in os.listdir('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/gt_npy'):
#     noisy_png_names = [
#         gt_npy_name.split('.')[0] + f'_sigma_{sigma}.png' for sigma in [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]
#     ]
#     noisy_npy_names = [
#         gt_npy_name.split('.')[0] + f'_sigma_{sigma}.npy' for sigma in [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]
#     ]
#     noisy_png_dir = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/noisy_png'
#     noisy_npy_dir = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/noisy_npy'
#     for noisy_png_name in noisy_png_names:
#         if not os.path.isfile(os.path.join(noisy_png_dir, noisy_png_name)):
#             print(gt_npy_name)
#     for noisy_npy_name in noisy_npy_names:
#         if not os.path.isfile(os.path.join(noisy_npy_dir, noisy_npy_name)):
#             print(gt_npy_name)

target_png_paths=[]
for gt_png_name in os.listdir('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/gt_png'):
    gt_npy_path = '/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/gt_npy/'+gt_png_name.replace('.png','.npy')
    if os.path.isfile(gt_npy_path):
        target_png_paths.append(os.path.join('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/gt_png',gt_png_name))

for p in target_png_paths:
    shutil.copyfile(p, p.replace('/data/','/data_zzy_orig/'))
