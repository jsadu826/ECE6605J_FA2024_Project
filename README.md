# Improved BM3D with Affine Transformation and Deep Learning-Based Filtering

- BM3D related files are in [bm3d](bm3d).
  - This folder is modified from [this repository](https://github.com/Ryanshuai/BM3D_py/blob/master/test_data).
  - Use [this file](bm3d/precompute_BM_zzy.py) to generate coarsely denoised image and matched blocks for deep learning training.

- Affine transform related files are in [affine_transform](affine_transform).

- Deep learning related files are in [deep_learning](deep_learning).
  - The SFCM module is defined [here](deep_learning/nbnet++/nbnet_final.py)
  - To train NBNET + SFCM (+ edge loss), modify codes in [this script](deep_learning/nbnet++/main.sh) and run.
  - To test NBNET + SFCM (+ edge loss), modify codes in [this script](deep_learning/nbnet++/test.sh) and run.
