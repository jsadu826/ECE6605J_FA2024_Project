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
    parser.add_argument('--tr_noisy_dir', type=str)
    parser.add_argument('--val_gt_dir', type=str)
    parser.add_argument('--val_noisy_dir', type=str)
    parser.add_argument('--ts_gt_dir', type=str)
    parser.add_argument('--ts_noisy_dir', type=str)
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
    parser.add_argument('--save_denoised_dir', type=str, default=None)

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.tr_gt_dir = args.tr_gt_dir
        self.tr_noisy_dir = args.tr_noisy_dir
        self.val_gt_dir = args.val_gt_dir
        self.val_noisy_dir = args.val_noisy_dir
        self.ts_gt_dir = args.ts_gt_dir
        self.ts_noisy_dir = args.ts_noisy_dir
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
        self.save_denoised_dir = args.save_denoised_dir


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


class Bm3dDataset(Dataset):
    def __init__(self, gt_npy_dir, noisy_npy_dir, is_train):
        super().__init__()
        self.gt_npy_dir = gt_npy_dir
        self.noisy_npy_dir = noisy_npy_dir
        self.is_train = is_train

        self.noisy_npy_paths = [os.path.join(noisy_npy_dir, name) for name in os.listdir(noisy_npy_dir)]

    def __len__(self):
        return len(self.noisy_npy_paths)

    def __getitem__(self, idx):
        gt_img = np.load(os.path.join(self.gt_npy_dir, os.path.basename(self.noisy_npy_paths[idx]).split('_')[0] + '.npy'))
        with open(self.noisy_npy_paths[idx], 'rb') as f:
            noisy_npy = pickle.load(f)
        sigma = int(self.noisy_npy_paths[idx].split('_')[-1][:-4])
        if sigma == 2:
            pass
        elif sigma == 50:
            sigma = np.random.randint(43, 58)
        elif sigma == 75:
            sigma = np.random.randint(65, 86)
        elif sigma == 100:
            sigma = np.random.randint(90, 111)
        else:
            sigma = np.random.randint(sigma - 2, sigma + 3)

        noisy_img = normalize_to_zero_one(torch.from_numpy(add_gaussian_noise(gt_img, sigma)).unsqueeze_(0).to(torch.float32))
        gt_img = normalize_to_zero_one(torch.from_numpy(gt_img).unsqueeze_(0).to(torch.float32))
        basic_img = normalize_to_zero_one(torch.from_numpy(noisy_npy['basic_img']).unsqueeze_(0).to(torch.float32))
        relation_2d_indexed = torch.from_numpy(noisy_npy['relation_2d_indexed']).to(torch.int16)
        relation_1d_indexed = torch.from_numpy(noisy_npy['relation_1d_indexed']).to(torch.int16)
        similar_count_2d_indexed = torch.from_numpy(noisy_npy['similar_count_2d_indexed']).to(torch.int16)
        similar_count_1d_indexed = torch.from_numpy(noisy_npy['similar_count_1d_indexed']).to(torch.int16)

        if self.is_train:
            # i, j, th, tw = RandomCrop.get_params(gt_img, (128, 128))
            # gt_img = T.crop(gt_img, i, j, th, tw)
            # noisy_img = T.crop(noisy_img, i, j, th, tw)
            # basic_img = T.crop(basic_img, i, j, th, tw)
            if torch.rand(1) < 0.5:
                gt_img = T.hflip(gt_img)
                noisy_img = T.hflip(noisy_img)
                basic_img = T.hflip(basic_img)
            if torch.rand(1) < 0.5:
                gt_img = T.vflip(gt_img)
                noisy_img = T.vflip(noisy_img)
                basic_img = T.vflip(basic_img)

        return gt_img, noisy_img, basic_img, relation_2d_indexed, relation_1d_indexed, similar_count_2d_indexed, similar_count_1d_indexed


class Bm3dTestDataset(Dataset):
    def __init__(self, gt_npy_dir, noisy_npy_dir, sigma):
        super().__init__()
        self.gt_npy_dir = gt_npy_dir
        self.noisy_npy_dir = noisy_npy_dir
        npy_names = [
            'Cameraman_sigma_2.npy',
            'house_sigma_2.npy',
            'Peppers_sigma_2.npy',
            'montage_sigma_2.npy',
            'Lena_sigma_2.npy',
            'barbara_sigma_2.npy',
            'boat_sigma_2.npy',
            'fingerprint_sigma_2.npy',
            'Man_sigma_2.npy',
            'couple_sigma_2.npy',
            'hill_sigma_2.npy'
        ]
        self.noisy_npy_paths = [os.path.join(noisy_npy_dir, name) for name in npy_names]
        self.sigma = sigma

    def __len__(self):
        return len(self.noisy_npy_paths)

    def __getitem__(self, idx):
        gt_img = np.load(os.path.join(self.gt_npy_dir, os.path.basename(self.noisy_npy_paths[idx]).split('_')[0] + '.npy'))
        with open(self.noisy_npy_paths[idx], 'rb') as f:
            noisy_npy = pickle.load(f)

        noisy_img = normalize_to_zero_one(torch.from_numpy(add_gaussian_noise(gt_img, self.sigma)).unsqueeze_(0).to(torch.float32))
        gt_img = normalize_to_zero_one(torch.from_numpy(gt_img).unsqueeze_(0).to(torch.float32))
        basic_img = normalize_to_zero_one(torch.from_numpy(noisy_npy['basic_img']).unsqueeze_(0).to(torch.float32))
        relation_2d_indexed = torch.from_numpy(noisy_npy['relation_2d_indexed']).to(torch.int16)
        relation_1d_indexed = torch.from_numpy(noisy_npy['relation_1d_indexed']).to(torch.int16)
        similar_count_2d_indexed = torch.from_numpy(noisy_npy['similar_count_2d_indexed']).to(torch.int16)
        similar_count_1d_indexed = torch.from_numpy(noisy_npy['similar_count_1d_indexed']).to(torch.int16)

        return gt_img, noisy_img, basic_img, relation_2d_indexed, relation_1d_indexed, similar_count_2d_indexed, similar_count_1d_indexed, os.path.basename(self.noisy_npy_paths[idx]).split('_')[
            0] + '.png'


def add_gaussian_noise(im, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    im = im + (sigma * np.random.randn(*im.shape)).astype(np.int16)
    im = np.clip(im, 0., 255., out=None)
    im = im.astype(np.uint8)
    return im


import torch
import torch.nn.functional as F


def sobel_operator(image):
    """
    Apply the Sobel operator to an image to detect edges.

    Parameters:
    - image (Tensor): Input image tensor of shape (B, C, H, W), where
                      B is the batch size, C is the number of channels,
                      H is the height, and W is the width.

    Returns:
    - edge_magnitude (Tensor): Tensor of edge magnitudes of the same shape as the input image.
    """

    # Define Sobel kernels for horizontal and vertical gradients
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

    # Move kernels to the same device as the input image
    device = image.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    # Apply the Sobel kernels to the image using convolution
    grad_x = F.conv2d(image, sobel_x, padding=1)  # Gradient in the x direction
    grad_y = F.conv2d(image, sobel_y, padding=1)  # Gradient in the y direction

    # Compute the magnitude of the gradient (edge magnitude)
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    return edge_magnitude
