"""
Mean Teacher V5: Dual-Consistency Mean Teacher with Interpolation Regularization
================================================================================

Diagnosis of V1-V4 failure:
  All versions (enhanced, v2, v3, v4) only used AUGMENTATION consistency —
  "predictions should be invariant to image perturbations". This saturates
  because augmentation diversity is limited. Adding sharpening, confidence
  weighting, CutMix, dual CE+MSE, etc. are all variations on the SAME
  regularization axis → diminishing returns.

Solution: Add an ORTHOGONAL regularization signal via Interpolation Consistency
Training (ICT). Instead of perturbing one image, MIX two images and enforce
that the prediction on the mixture equals the mixture of predictions:
  f(λx₁ + (1-λ)x₂) ≈ λf(x₁) + (1-λ)f(x₂)

This directly smooths the decision boundary between training samples — a
fundamentally different inductive bias from augmentation invariance.

Changes from baseline (3 changes):
  1. Weak-to-strong augmentation consistency (FixMatch/UniMatch)
     + Fixed GaussNoise (std_range), added CLAHE + GridDistortion for
     colonoscopy domain (dark regions, lens distortion)
  2. Interpolation Consistency Training (ICT, Verma et al., Neural Networks 2022)
     MixUp within-batch on weak views → MSE on mixed predictions
  3. Dual consistency = augmentation MSE + interpolation MSE
     Two orthogonal signals: invariance (around points) + smoothness (between points)

Results from prior versions at 10% labels:
  - baseline mean_teacher:  Dice avg 0.8982
  - enhanced (CE+W2S):     Dice avg 0.9007
  - enhanced_v2 (MSE+CE):  Dice avg 0.8999
  - v3 (triple-stream):    Dice avg 0.8984
  - v4 (sharpened+balanced):Dice avg 0.8994
"""

import os
import time
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
import albumentations as A
from torchvision import transforms
from monai.losses import *
from monai.metrics import *

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed


# ============================================================
# CHANGE 1: Fixed & domain-specific augmentation
# ============================================================

class RobustStrongWeakAugment(StrongWeakAugment):
    """
    Fixes broken GaussNoise (var_limit → std_range) and adds
    colonoscopy-specific augmentations:
      - CLAHE: adaptive contrast for dark/underexposed regions
      - GridDistortion: simulates endoscope wide-angle lens distortion
    """
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super().__init__(dataset, img_size, use_aug, data_path)
        s_p = 0.7
        self.strong_augment = A.Compose([
            A.GaussNoise(std_range=(0.02, 0.1), p=s_p),          # fixed API
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),  # dark regions
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),   # lens effect
            A.ElasticTransform(alpha=40, sigma=4, p=s_p),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=s_p),
            A.GaussianBlur(blur_limit=(3, 7), p=s_p),
            A.MotionBlur(blur_limit=(3, 7), p=s_p),
        ])


# ============================================================
# Model utilities (unchanged from baseline)
# ============================================================

def create_model(ema=False):
    model = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


# ============================================================
# Main
# ============================================================

def main(config):
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=RobustStrongWeakAugment    # CHANGE 1: fixed augmentation
    )

    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True, num_workers=config.train.num_workers, pin_memory=True
    )
    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True, num_workers=config.train.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False, num_workers=config.test.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False, num_workers=config.test.num_workers, pin_memory=True
    )

    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    model = create_model()
    ema_model = create_model(ema=True)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total, {total_trainable/1e6:.2f}M trainable')

    model = model.cuda()
    ema_model = ema_model.cuda()

    criterion_sup = GeneralizedDiceFocalLoss(
        softmax=True, to_onehot_y=False, include_background=True
    ).cuda()
    criterion_mse = nn.MSELoss().cuda()

    best_model = train_val(config, model, ema_model, train_loader, val_loader,
                           criterion_sup, criterion_mse)
    test(config, best_model, best_model_dir, test_loader, criterion_sup)


def train_val(config, model, ema_model, train_loader, val_loader,
              criterion_sup, criterion_mse):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs, eta_min=1e-6
    )

    train_dice = DiceMetric(include_background=True, reduction="mean")

    early_stop_patience = config.train.early_stop_patience if getattr(config.train, 'early_stop_patience', None) is not None else 20
    early_stop_min_epochs = config.train.early_stop_min_epochs if getattr(config.train, 'early_stop_min_epochs', None) is not None else 50
    no_improve_epochs = 0
    max_dice = -float('inf')
    best_epoch = 0
    global_step = 0

    for epoch in range(config.train.num_epochs):
        start = time.time()
        model.train()
        ema_model.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0

        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        train_dice.reset()

        for batch_idx, (batch_l, batch_u) in enumerate(train_loop):
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()

            # Weak view for teacher, strong view for student
            img_u_w = batch_u['img_w'].cuda().float()
            img_u_s = batch_u['img_s'].cuda().float()

            B = img_u_w.shape[0]

            # =============================================
            # Teacher: predict on weak view (no gradient)
            # =============================================
            with torch.no_grad():
                teacher_probs = ema_model(img_u_w).softmax(dim=1)  # (B,C,H,W)

            # =============================================
            # CHANGE 2: ICT — MixUp within batch
            # =============================================
            lam = np.random.beta(args.ict_alpha, args.ict_alpha)
            lam = max(lam, 1 - lam)  # ensure dominant image is always first

            idx = torch.randperm(B, device=img_u_w.device)
            img_mix = lam * img_u_w + (1 - lam) * img_u_w[idx]

            with torch.no_grad():
                target_mix = lam * teacher_probs + (1 - lam) * teacher_probs[idx]

            # =============================================
            # Student forward passes
            # =============================================
            # Supervised
            outputs_l = model(img_l)
            sup_loss = criterion_sup(outputs_l, label_l)

            # Efficient: single forward for strong + mixed
            student_cat = model(torch.cat([img_u_s, img_mix], dim=0))
            student_s, student_mix = student_cat.chunk(2, dim=0)

            # CHANGE 1: Augmentation consistency — MSE(student_strong, teacher_weak)
            aug_loss = criterion_mse(student_s.softmax(dim=1), teacher_probs)

            # CHANGE 2: Interpolation consistency — MSE(student_mix, mixed_target)
            ict_loss = criterion_mse(student_mix.softmax(dim=1), target_mix)

            # =============================================
            # CHANGE 3: Dual consistency = aug + ict
            # =============================================
            consistency_weight = get_current_consistency_weight(epoch)
            loss = sup_loss + consistency_weight * (aug_loss + args.ict_weight * ict_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

            # Metrics
            with torch.no_grad():
                output_onehot = torch.zeros_like(outputs_l)
                output_onehot.scatter_(1, outputs_l.argmax(dim=1, keepdim=True), 1)
                train_dice(y_pred=output_onehot, y=label_l)
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * img_l.shape[0]) / (num_train + img_l.shape[0])
                num_train += img_l.shape[0]

            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}",
                'Aug': f"{aug_loss.item():.4f}",
                'ICT': f"{ict_loss.item():.4f}"
            })

            if config.debug:
                break

        train_metrics['dice'] = train_dice.aggregate().item()

        # Validation
        val_metrics = validate_model(ema_model, val_loader, criterion_sup)

        if val_metrics['dice'] > max_dice:
            max_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save(ema_model.state_dict(), best_model_dir)
            no_improve_epochs = 0
            message = (f'New best epoch {epoch}! '
                      f'Dice: {val_metrics["dice"]:.4f}, '
                      f'IoU: {val_metrics["iou"]:.4f}, '
                      f'HD: {val_metrics["hd"]:.4f}')
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        else:
            no_improve_epochs += 1

        if epoch >= early_stop_min_epochs and no_improve_epochs > early_stop_patience:
            early_message = f'Early stopping at epoch {epoch} after {no_improve_epochs} epochs without improvement.'
            print(early_message)
            file_log.write(early_message + '\n')
            file_log.flush()
            break

        scheduler.step()
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} | Train Dice: {train_metrics["dice"]:.4f} | '
              f'Val Dice: {val_metrics["dice"]:.4f} | '
              f'IoU: {val_metrics["iou"]:.4f} | HD: {val_metrics["hd"]:.4f} | '
              f'Time: {time_elapsed//60:.0f}m{time_elapsed%60:.0f}s')
        print('='*80)

        if config.debug:
            break

    print(f'Training completed. Best epoch: {best_epoch}')
    return ema_model


def validate_model(model, val_loader, criterion):
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)

    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)

            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)

            if len(label.shape) == 4:
                labels_onehot = label
            else:
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)

            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)

            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * img.shape[0]) / (num_val + img.shape[0])
            num_val += img.shape[0]

    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()

    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()

    return metrics


def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)

    results_str = (f"Test Results:\n"
                  f"Loss: {metrics['loss']:.4f}\n"
                  f"Dice: {metrics['dice']:.4f}\n"
                  f"IoU: {metrics['iou']:.4f}\n"
                  f"HD: {metrics['hd']:.4f}")

    with open(test_results_dir, 'w') as f:
        f.write(results_str)

    print('='*80)
    print(results_str)
    print('='*80)

    file_log.write('\n' + '='*80 + '\n')
    file_log.write(results_str + '\n')
    file_log.write('='*80 + '\n')
    file_log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mean Teacher V5: ICT-Enhanced Dual Consistency')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    # V5 hyperparameters
    parser.add_argument('--ict_alpha', type=float, default=1.0,
                        help='Beta distribution param for MixUp lambda. 1.0=uniform, 0.2=near-extremes')
    parser.add_argument('--ict_weight', type=float, default=1.0,
                        help='Weight of ICT loss relative to augmentation consistency loss')
    parser.add_argument('--folds', type=int, nargs='+', default=[1,2,3,4,5],
                        help='Which folds to run (e.g. --folds 1 for tuning)')

    args = parser.parse_args()

    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")

    store_config = config
    config = DotDict(config)

    for fold in args.folds:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold

        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f'{exp_dir}/best.pth'
        test_results_dir = f'{exp_dir}/test_results.txt'

        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))

        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            main(config)

        torch.cuda.empty_cache()
