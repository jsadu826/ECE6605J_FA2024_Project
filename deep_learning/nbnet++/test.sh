#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --ts_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/test_bm3d_images/gt_npy \
    --ts_noisy_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/test_bm3d_images/noisy_npy \
    --model_file /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/nbnet++/runs/20241218_062309_edge_loss/best_model_ep118_psnr30.590019_ssim0.847561 \
    --save_denoised_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/nbnet++/test_edge_loss
