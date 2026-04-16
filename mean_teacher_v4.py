"""
Mean Teacher V4: Sharpened Confidence-Weighted Mean Teacher (SCWMT)
==================================================================
4 targeted changes from vanilla Mean Teacher baseline.

Baseline analysis (10% labels):
  - mean_teacher_10p (MSE, same view):     Dice avg 0.8982
  - mt_enhanced_10p (CE, weak→strong):     Dice avg 0.9007  (best @10%)
  - mt_enhanced_v2_10p (MSE+CE, CutMix):   Dice avg 0.8999

Key insights from prior experiments:
  1. MSE soft-consistency > CE hard pseudo-labels on this small dataset
  2. Weak-to-strong paradigm adds consistent value (+0.25%)
  3. Hard confidence threshold (0.95) discards useful pixels
  4. Background pixels dominate MSE → polyp learning signal diluted

Research-backed changes:
  1. Weak-to-Strong paradigm (FixMatch/UniMatch)
     Teacher sees weak aug → student sees strong aug → learns harder features
  2. Temperature Sharpening on teacher predictions (MixMatch, NeurIPS'19)
     Soft targets become more decisive without losing distribution info
  3. Confidence-Weighted MSE (SoftMatch/FreeMatch, ICLR'23)
     Smooth pixel weighting by max(teacher_prob)^α — no hard threshold
  4. Class-Balanced Consistency (domain-specific for polyp segmentation)
     Inverse-frequency reweighting prevents background domination in MSE
"""

import os
import time
import argparse
from datetime import datetime
import cv2

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
from monai.losses import *
from monai.metrics import *

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed


# ============================================================
# V4 Components
# ============================================================

def sharpen(probs, temperature=0.5):
    """
    Temperature sharpening on probability distribution (MixMatch, NeurIPS'19).
    Lower temperature → sharper (more confident) predictions.
    T=1.0 → no change, T→0 → argmax.
    """
    probs_sharp = probs ** (1.0 / temperature)
    return probs_sharp / (probs_sharp.sum(dim=1, keepdim=True) + 1e-8)


def compute_confidence_weight(probs, alpha=2.0):
    """
    Smooth confidence weighting based on max probability (SoftMatch-inspired).
    High confidence → weight ≈ 1, low confidence → weight → 0.
    alpha controls sharpness: higher alpha = more aggressive downweighting.
    """
    conf = probs.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
    return conf ** alpha


def compute_class_balance_weight(probs, max_weight=5.0):
    """
    Inverse-frequency reweighting based on teacher's predicted class distribution.
    Prevents background from dominating MSE consistency loss.
    """
    B, C, H, W = probs.shape
    pred_classes = probs.argmax(dim=1)  # (B, H, W)

    class_counts = torch.zeros(C, device=probs.device)
    for c in range(C):
        class_counts[c] = (pred_classes == c).sum().float() + 1.0
    median_count = class_counts.median()
    class_weights = (median_count / class_counts).clamp(min=0.5, max=max_weight)

    # Map per-pixel
    pixel_weight = class_weights[pred_classes]  # (B, H, W)
    return pixel_weight.unsqueeze(1)  # (B, 1, H, W)


# ============================================================
# Model utilities
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
    # CHANGE 1: StrongWeakAugment for weak-to-strong paradigm
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment       # teacher=weak, student=strong
    )

    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )

    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    model = create_model()
    ema_model = create_model(ema=True)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model = model.cuda()
    ema_model = ema_model.cuda()

    criterion_sup = GeneralizedDiceFocalLoss(
        softmax=True,
        to_onehot_y=False,
        include_background=True
    ).cuda()

    # No separate consistency criterion needed — V4 computes weighted MSE inline

    best_model = train_val(config, model, ema_model, train_loader, val_loader, criterion_sup)
    test(config, best_model, best_model_dir, test_loader, criterion_sup)


def train_val(config, model, ema_model, train_loader, val_loader, criterion_sup):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs, eta_min=1e-6)

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

            # CHANGE 1: Weak-to-strong — teacher sees weak, student sees strong
            img_u_w = batch_u['img_w'].cuda().float()   # weak → teacher
            img_u_s = batch_u['img_s'].cuda().float()   # strong → student

            # Forward passes
            outputs_l = model(img_l)
            outputs_u_s = model(img_u_s)           # student on strong view

            with torch.no_grad():
                teacher_out = ema_model(img_u_w)    # teacher on weak view
                teacher_probs = teacher_out.softmax(dim=1)

                # CHANGE 2: Temperature sharpening
                teacher_sharp = sharpen(teacher_probs, temperature=args.temperature)

                # CHANGE 3: Confidence-weighted pixel importance
                conf_weight = compute_confidence_weight(teacher_probs, alpha=args.conf_alpha)

                # CHANGE 4: Class-balanced reweighting
                class_weight = compute_class_balance_weight(teacher_probs, max_weight=args.class_max_weight)

                # Combined weight map  (B, 1, H, W)
                pixel_weight = conf_weight * class_weight

            # Supervised loss (unchanged from baseline)
            sup_loss = criterion_sup(outputs_l, label_l)

            # V4 Consistency loss: Sharpened Confidence-Weighted Class-Balanced MSE
            student_probs = outputs_u_s.softmax(dim=1)
            mse_per_pixel = (student_probs - teacher_sharp) ** 2   # (B, C, H, W)
            consistency_loss = (pixel_weight * mse_per_pixel).mean()

            consistency_weight = get_current_consistency_weight(epoch)
            loss = sup_loss + consistency_weight * consistency_loss

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

            # Progress bar with V4-specific info
            avg_conf = conf_weight.mean().item()
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}",
                'Conf': f"{avg_conf:.2f}"
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
    parser = argparse.ArgumentParser(description='Mean Teacher V4 — SCWMT')
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
    # V4 hyperparameters
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sharpening temperature (lower = sharper). T=1 means no sharpening.')
    parser.add_argument('--conf_alpha', type=float, default=2.0,
                        help='Exponent for confidence weighting. Higher = more aggressive downweighting.')
    parser.add_argument('--class_max_weight', type=float, default=5.0,
                        help='Max class-balance weight (caps minority class boost)')

    args = parser.parse_args()

    # Load and update config
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold

    # Setup CUDA and seeds
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # Print configuration
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")

    store_config = config
    config = DotDict(config)

    for fold in [1, 2, 3, 4, 5]:
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
