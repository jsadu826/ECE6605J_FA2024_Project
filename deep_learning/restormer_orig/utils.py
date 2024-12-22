import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_gt_dir', type=str)
    parser.add_argument('--val_gt_dir', type=str)
    parser.add_argument('--ts_gt_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8])  # num transformer blocks
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--channels', nargs='+', type=int, default=[48, 96, 192, 384])
    parser.add_argument('--expansion_factor', type=float, default=2.66)  # GDFN channel expansion factor
    parser.add_argument('--num_refinement', type=int, default=4)  # num channels at refinement stage

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1)  # -1 means no manual seed
    parser.add_argument('--model_file', type=str, default=None)  # model_file is None means training stage, else means testing stage

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.tr_gt_dir = args.tr_gt_dir
        self.val_gt_dir = args.val_gt_dir
        self.ts_gt_dir = args.ts_gt_dir
        self.save_dir = args.save_dir

        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement

        self.batch_size = args.batch_size
        self.lr = args.lr

        self.num_epochs = args.num_epochs
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.model_file = args.model_file


def init_args(args):
    if not args.model_file:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        if args.seed >= 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False

    return Config(args)


def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()


def normalize_to_zero_one(x: torch.Tensor):
    return ((x - x.min()) / (x.max() - x.min())).clip_(0, 1)


def add_gaussian_noise(im, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = im + (sigma * np.random.randn(*im.shape)).astype(np.int16)
    im = np.clip(im, 0., 255., out=None)
    im = im.astype(np.uint8)
    return im


class Bm3dDatasetForRepro(Dataset):
    def __init__(self, gt_npy_dir, is_train):
        super().__init__()
        self.gt_npy_dir = gt_npy_dir
        self.gt_npy_paths = [os.path.join(gt_npy_dir, name) for name in os.listdir(gt_npy_dir)]
        self.is_train = is_train

    def __len__(self):
        return len(self.gt_npy_paths)

    def __getitem__(self, idx):
        gt_img = np.load(self.gt_npy_paths[idx])
        sigma = random.choice([2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100])
        noisy_img = add_gaussian_noise(gt_img, sigma)

        gt_img = normalize_to_zero_one(torch.from_numpy(gt_img).unsqueeze_(0).to(torch.float32))
        noisy_img = normalize_to_zero_one(torch.from_numpy(noisy_img).unsqueeze_(0).to(torch.float32))

        if self.is_train:
            i, j, th, tw = RandomCrop.get_params(gt_img, (128,128))
            gt_img = T.crop(gt_img, i, j, th, tw)
            noisy_img = T.crop(noisy_img, i, j, th, tw)
            if torch.rand(1) < 0.5:
                gt_img = T.hflip(gt_img)
                noisy_img = T.hflip(noisy_img)
            if torch.rand(1) < 0.5:
                gt_img = T.vflip(gt_img)
                noisy_img = T.vflip(noisy_img)

        return gt_img, noisy_img
