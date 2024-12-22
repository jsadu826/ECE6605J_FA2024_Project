#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
save_dir=runs/$now
mkdir -p "$save_dir"
CUDA_VISIBLE_DEVICES=1 python main.py \
    --tr_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/train/gt_npy \
    --val_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/valid/gt_npy \
    --ts_gt_dir /data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/data/test/gt_npy \
    --save_dir "$save_dir" \
    --batch_size 8 \
    --lr 1e-4 \
    --num_epochs 500 \
    --seed 233333 \
