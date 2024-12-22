#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
save_dir=runs/"$now"_edge_loss
mkdir -p "$save_dir"
CUDA_VISIBLE_DEVICES=3 python main.py \
    --tr_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/gt_npy \
    --tr_noisy_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/noisy_npy \
    --val_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/valid/gt_npy \
    --val_noisy_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/valid/noisy_npy \
    --ts_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/test/gt_npy \
    --ts_noisy_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/test/noisy_npy \
    --save_dir "$save_dir" \
    --batch_size 48 \
    --lr 3e-4 \
    --num_epochs 500 \
    --seed 233333 \
