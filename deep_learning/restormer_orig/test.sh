#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_file /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/restormer_orig/runs/20241128_003434/best_model_ep153_psnr32.278822_ssim0.879748
