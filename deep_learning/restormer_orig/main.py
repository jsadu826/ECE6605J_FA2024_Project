import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from restormer import Restormer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Bm3dDatasetForRepro, parse_args, psnr, ssim


def eval_loop(model, data_loader, epoch):
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        eval_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for (gt_img, noisy_img) in eval_bar:
            gt_img = gt_img.cuda()
            noisy_img = noisy_img.cuda()
            print('data shapes:', gt_img.shape, noisy_img.shape)

            denoised_img = model(noisy_img)

            gt_img = torch.clamp(gt_img.mul(255), 0, 255)
            noisy_img = torch.clamp(noisy_img.mul(255), 0, 255)
            denoised_img = torch.clamp((torch.clamp(denoised_img, 0, 1).mul(255)), 0, 255)

            current_psnr, current_ssim = psnr(denoised_img, gt_img), ssim(denoised_img, gt_img)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1

            Image.fromarray(gt_img[0, 0].detach().contiguous().cpu().numpy().astype(np.uint8), mode='L').save(os.path.join(args.save_dir, 'eval_gt.png'))
            Image.fromarray(noisy_img[0, 0].detach().contiguous().cpu().numpy().astype(np.uint8), mode='L').save(os.path.join(args.save_dir, 'eval_noisy.png'))
            Image.fromarray(denoised_img[0, 0].detach().contiguous().cpu().numpy().astype(np.uint8), mode='L').save(os.path.join(args.save_dir, 'eval_denoised.png'))

            eval_bar.set_description('eval epoch: {}, PSNR: {:.2f}, SSIM: {:.3f}'.format(epoch, total_psnr / count, total_ssim / count))

    return total_psnr / count, total_ssim / count


if __name__ == '__main__':
    args = parse_args()

    model = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).cuda()

    if args.model_file:
        def add_gaussian_noise(im, sigma, seed=None):
            if seed is not None:
                np.random.seed(seed)
            im = im + (sigma * np.random.randn(*im.shape)).astype(np.int16)
            im = np.clip(im, 0., 255., out=None)
            im = im.astype(np.uint8)
            return im

        model.load_state_dict(torch.load(args.model_file), strict=True)

        with torch.no_grad():
            img_names = ['Cameraman.png', 'house.png', 'Peppers.png', 'montage.png', 'Lena.png', 'barbara.png', 'boat.png', 'fingerprint.png', 'Man.png', 'couple.png', 'hill.png']
            sigmas = [2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]

            for i, sigma in enumerate(sigmas):
                save_denoised_dir = f'/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/restormer_orig/denoised_images/sigma_{sigma}'
                os.makedirs(save_denoised_dir, exist_ok=True)
                print(sigma, end=' ')
                for img_name in img_names:
                    clean_orig = np.array(cv2.imread('/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/clean/' + img_name, cv2.IMREAD_GRAYSCALE))
                    noisy_orig = np.array(cv2.imread(f'/data2/zheyu_data/workspace/denoise/BM3D-github/deep_learning/bm3d_images/sigma_{sigma}/' + img_name, cv2.IMREAD_GRAYSCALE))

                    clean_tensor = torch.from_numpy(clean_orig).unsqueeze(0).unsqueeze(0).to(torch.float32).cuda()
                    noisy_tensor = torch.from_numpy(noisy_orig).unsqueeze(0).unsqueeze(0).to(torch.float32).cuda()

                    noisy_tensor_normed = ((noisy_tensor - noisy_tensor.min()) / (noisy_tensor.max() - noisy_tensor.min())).clip_(0, 1)

                    denoised_tensor_normed = model(noisy_tensor_normed)
                    denoised_tensor = denoised_tensor_normed.clip_(0, 1) * 255
                    Image.fromarray(denoised_tensor.cpu().type(torch.uint8)[0, 0].numpy(), mode='L').save(os.path.join(save_denoised_dir, img_name))

                    current_psnr, current_ssim = psnr(denoised_tensor, clean_tensor), ssim(denoised_tensor, clean_tensor)
                    current_psnr, current_ssim = round(float(current_psnr), 2), round(float(current_ssim * 100), 2)

                    print('& \\textcolor{{blue}}{{{}}}/\\textcolor{{red}}{{{}}}'.format(current_psnr, current_ssim), end=' ')
                print('\\\\ \\hline')

        exit()

    train_dataset = Bm3dDatasetForRepro(args.tr_gt_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_dataset = Bm3dDatasetForRepro(args.val_gt_dir, is_train=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    train_bar = tqdm(range(1, args.num_epochs + 1), initial=1, dynamic_ncols=True)
    writer = SummaryWriter(log_dir=args.save_dir)
    best_psnr, best_ssim = 0.0, 0.0

    for epoch in train_bar:
        total_loss, total_num = 0.0, 0

        for idx, (gt_img, noisy_img) in enumerate(train_loader):
            gt_img = gt_img.cuda()
            noisy_img = noisy_img.cuda()
            print('data shapes:', gt_img.shape, noisy_img.shape)

            denoised_tensor = model(noisy_img)
            loss = F.l1_loss(denoised_tensor, gt_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += noisy_img.shape[0]
            total_loss += loss.item() * noisy_img.shape[0]
            train_bar.set_description('train epoch: {}, batch: {}, epoch_avg_loss: {:.3f}'.format(epoch, idx + 1, total_loss / total_num))

            gt_img = torch.clamp(gt_img[:1].mul(255), 0, 255)
            noisy_img = torch.clamp(noisy_img[:1].mul(255), 0, 255)
            denoised_tensor = torch.clamp((torch.clamp(denoised_tensor[:1], 0, 1).mul(255)), 0, 255)
            Image.fromarray(gt_img[0, 0].detach().contiguous().cpu().numpy().astype(np.uint8), mode='L').save(os.path.join(args.save_dir, 'train_gt.png'))
            Image.fromarray(noisy_img[0, 0].detach().contiguous().cpu().numpy().astype(np.uint8), mode='L').save(os.path.join(args.save_dir, 'train_noisy.png'))
            Image.fromarray(denoised_tensor[0, 0].detach().contiguous().cpu().numpy().astype(np.uint8), mode='L').save(os.path.join(args.save_dir, 'train_denoised.png'))

        lr_scheduler.step()
        writer.add_scalar('train_loss', total_loss / total_num, epoch)

        epoch_psnr, epoch_ssim = eval_loop(model, valid_loader, epoch)
        writer.add_scalar('eval_psnr', epoch_psnr, epoch)
        writer.add_scalar('eval_ssim', epoch_ssim, epoch)

        if epoch_psnr > best_psnr and epoch_ssim > best_ssim:
            best_psnr = epoch_psnr
            best_ssim = epoch_ssim
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_ep{}_psnr{:2f}_ssim{:3f}'.format(epoch, best_psnr, best_ssim)))
