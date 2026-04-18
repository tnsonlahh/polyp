"""
Mean Teacher V6: IAC-Enhanced Dual-Consistency (V5 + Illumination-Aware Consistency)
======================================================================================

Motivation
----------
V5 established dual consistency: augmentation invariance (aug_loss) + boundary
smoothness via ICT (ict_loss). Both are orthogonal regularization axes, which is
why V5 outperforms V1-V4 (single axis).

V6 replaces the plain MSE augmentation consistency in V5 with IAC
(Illumination-Aware Consistency from IDA-MT):

  Plain aug MSE:  MSE(f_student(x_strong), f_teacher(x_weak))
  IAC aug loss:   Σ_xy  w(x,y) · MSE_xy(...)
                  w = dark_boost × entropy_gate

Rationale:
  - Colonoscopy images have strong radial illumination gradient (center bright,
    corners dark). The teacher systematically under-segments polyps in dark
    regions early in training → feeding those pixels at uniform weight into
    MSE amplifies noise.
  - IAC upweights dark pixels that need more learning signal, while gating
    out pixels where the teacher is uncertain (high entropy → noisy pseudo-label).
  - IAC warmup (linear 0→1 over iac_rampup epochs) prevents confirmation bias
    before the teacher has stabilized.
  - ICT is kept unchanged — it provides the orthogonal boundary-smoothness
    signal that made V5 beat V1-V4.

Loss:
    L = L_sup(labeled)
        + λ_c(t) · [ L_IAC(student_strong, teacher_weak, img_weak)   ← improved aug axis
                    + w_ict · L_ICT(student_mix, teacher_mix) ]       ← orthogonal axis

Changes from V5 (minimal):
  1. aug_loss = iac_loss(...) instead of MSE
  2. Linear IAC warmup (iac_rampup epochs)
  3. Two new args: --iac_alpha, --iac_beta, --iac_rampup
  4. Early stop re-enabled (was disabled in IDA-MT experiment)

Results target: beat V5 avg Dice 0.9029 at 10% labels.
"""

import os
import time
import argparse

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
from monai.losses import *
from monai.metrics import *

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed


# ============================================================
# Augmentation — same as V5 (domain-specific, strong)
# ============================================================

class RobustStrongWeakAugment(StrongWeakAugment):
    """V5 augmentation unchanged — strong aug is the source of aug_loss signal."""
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super().__init__(dataset, img_size, use_aug, data_path)
        s_p = 0.7
        self.strong_augment = A.Compose([
            A.GaussNoise(std_range=(0.02, 0.1), p=s_p),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.ElasticTransform(alpha=40, sigma=4, p=s_p),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=s_p),
            A.GaussianBlur(blur_limit=(3, 7), p=s_p),
            A.MotionBlur(blur_limit=(3, 7), p=s_p),
        ])


# ============================================================
# IAC: Illumination-Aware Consistency (from IDA-MT)
# ============================================================

def compute_luminance(img: torch.Tensor) -> torch.Tensor:
    """
    Rec.601 luminance, smoothed with 15×15 avg pool, per-image min-max normalized.
    Returns: (B, 1, H, W) in [0, 1] where 0=darkest region, 1=brightest.
    """
    L = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    L = F.avg_pool2d(L, kernel_size=15, stride=1, padding=7)
    B = L.shape[0]
    L_flat = L.view(B, -1)
    L_min = L_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    L_max = L_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (L - L_min) / (L_max - L_min + 1e-6)


def iac_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor,
             img_weak: torch.Tensor,
             alpha: float = 2.0, beta: float = 2.0,
             strength: float = 1.0) -> torch.Tensor:
    """
    Illumination-Aware Consistency loss.

        w(x,y) = blend( 1,  (1 + α(1-L)) · exp(-β·H_norm) )
                        ↑ plain MSE      ↑ IAC weighting
        blend weight = strength (0→1 via linear warmup)

    Args:
        student_logits: (B, C, H, W) student output (pre-softmax)
        teacher_probs:  (B, C, H, W) teacher softmax (detached)
        img_weak:       (B, 3, H, W) weak-view image (for luminance)
        alpha:          dark-boost strength, w_dark ∈ [1, 1+α]
        beta:           entropy gate steepness
        strength:       IAC warmup scalar ∈ [0, 1]
    """
    student_probs = student_logits.softmax(dim=1)
    sq_err = (student_probs - teacher_probs).pow(2).mean(dim=1, keepdim=True)

    # Dark boost
    L = compute_luminance(img_weak)
    dark_boost = 1.0 + alpha * (1.0 - L)

    # Entropy gate
    C = teacher_probs.shape[1]
    H = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=1, keepdim=True)
    gate = torch.exp(-beta * H / np.log(C))

    w = dark_boost * gate

    # Blend: strength=0 → uniform weight (vanilla MSE), strength=1 → full IAC
    w_blend = (1.0 - strength) * torch.ones_like(w) + strength * w
    # Normalize to keep loss magnitude stable across warmup
    w_blend = w_blend / (w_blend.mean(dim=(2, 3), keepdim=True) + 1e-6)

    return (w_blend * sq_err).mean()


# ============================================================
# Model utilities
# ============================================================

def create_model(ema=False):
    model = deeplabv3plus_resnet101(num_classes=2, output_stride=8, pretrained_backbone=True)
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


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    return float(np.clip(current / rampup_length, 0.0, 1.0))


def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def get_iac_strength(epoch):
    return linear_rampup(epoch, args.iac_rampup)


# ============================================================
# Main
# ============================================================

def main(config):
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.1),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=RobustStrongWeakAugment
    )

    l_train_loader = DataLoader(
        dataset['lb_dataset'], batch_size=config.train.l_batchsize,
        shuffle=True, num_workers=config.train.num_workers, pin_memory=True
    )
    u_train_loader = DataLoader(
        dataset['ulb_dataset'], batch_size=config.train.u_batchsize,
        shuffle=True, num_workers=config.train.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        dataset['val_dataset'], batch_size=config.test.batch_size,
        shuffle=False, num_workers=config.test.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        dataset['val_dataset'], batch_size=config.test.batch_size,
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

    early_stop_patience = getattr(config.train, 'early_stop_patience', 20)
    early_stop_min_epochs = getattr(config.train, 'early_stop_min_epochs', 50)
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

        consistency_weight = get_current_consistency_weight(epoch)
        iac_strength = get_iac_strength(epoch)

        for batch_idx, (batch_l, batch_u) in enumerate(train_loop):
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()
            img_u_w = batch_u['img_w'].cuda().float()
            img_u_s = batch_u['img_s'].cuda().float()

            B = img_u_w.shape[0]

            # --- Teacher on weak view ---
            with torch.no_grad():
                teacher_probs = ema_model(img_u_w).softmax(dim=1)

            # --- ICT: MixUp within batch ---
            lam = np.random.beta(args.ict_alpha, args.ict_alpha)
            lam = max(lam, 1 - lam)
            idx = torch.randperm(B, device=img_u_w.device)
            img_mix = lam * img_u_w + (1 - lam) * img_u_w[idx]
            with torch.no_grad():
                target_mix = lam * teacher_probs + (1 - lam) * teacher_probs[idx]

            # --- Student forward: supervised + strong + mixed (batched) ---
            outputs_l = model(img_l)
            sup_loss = criterion_sup(outputs_l, label_l)

            student_cat = model(torch.cat([img_u_s, img_mix], dim=0))
            student_s, student_mix = student_cat.chunk(2, dim=0)

            # --- IAC augmentation consistency (replaces plain MSE from V5) ---
            aug_loss = iac_loss(
                student_s, teacher_probs, img_u_w,
                alpha=args.iac_alpha, beta=args.iac_beta,
                strength=iac_strength
            )

            # --- ICT interpolation consistency (unchanged from V5) ---
            ict_loss_val = criterion_mse(student_mix.softmax(dim=1), target_mix)

            # --- Total loss ---
            loss = sup_loss + consistency_weight * (aug_loss + args.ict_weight * ict_loss_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

            with torch.no_grad():
                output_onehot = torch.zeros_like(outputs_l)
                output_onehot.scatter_(1, outputs_l.argmax(dim=1, keepdim=True), 1)
                train_dice(y_pred=output_onehot, y=label_l)
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * img_l.shape[0]) / (num_train + img_l.shape[0])
                num_train += img_l.shape[0]

            train_loop.set_postfix({
                'L': f"{loss.item():.3f}",
                'Sup': f"{sup_loss.item():.3f}",
                'IAC': f"{aug_loss.item():.3f}",
                'ICT': f"{ict_loss_val.item():.3f}",
                'iac_s': f"{iac_strength:.2f}",
                'Dice': f"{train_dice.aggregate().item():.3f}"
            })

            if config.debug:
                break

        train_metrics['dice'] = train_dice.aggregate().item()

        # --- Validation ---
        val_metrics = validate_model(ema_model, val_loader, criterion_sup)

        if val_metrics['dice'] > max_dice:
            max_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save(ema_model.state_dict(), best_model_dir)
            no_improve_epochs = 0
            message = (f'[New best] Ep {epoch} | Dice {val_metrics["dice"]:.4f} '
                       f'| IoU {val_metrics["iou"]:.4f} | HD95 {val_metrics["hd"]:.4f}')
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        else:
            no_improve_epochs += 1

        if epoch >= early_stop_min_epochs and no_improve_epochs > early_stop_patience:
            msg = f'Early stop at epoch {epoch} (no improvement for {no_improve_epochs} epochs).'
            print(msg)
            file_log.write(msg + '\n')
            file_log.flush()
            break

        scheduler.step()
        dt = time.time() - start
        print(f'Ep {epoch} | Train D {train_metrics["dice"]:.4f} | Val D {val_metrics["dice"]:.4f} '
              f'| IoU {val_metrics["iou"]:.4f} | HD {val_metrics["hd"]:.4f} '
              f'| iac_s {iac_strength:.2f} | {dt//60:.0f}m{dt%60:.0f}s')
        print('=' * 80)

        if config.debug:
            break

    print(f'Done. Best epoch: {best_epoch} (Dice={max_dice:.4f})')
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

    results_str = (f"Test Results:\nLoss: {metrics['loss']:.4f}\n"
                   f"Dice: {metrics['dice']:.4f}\nIoU: {metrics['iou']:.4f}\nHD: {metrics['hd']:.4f}")

    with open(test_results_dir, 'w') as f:
        f.write(results_str)

    print('=' * 80)
    print(results_str)
    print('=' * 80)
    file_log.write('\n' + '=' * 80 + '\n' + results_str + '\n' + '=' * 80 + '\n')
    file_log.flush()


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mean Teacher V6: IAC + ICT Dual Consistency')
    parser.add_argument('--exp', type=str, default='mt_v6')
    parser.add_argument('--config_yml', type=str, default='Configs/kvasir_seg.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='kvasir')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)

    # Mean Teacher core
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # ICT (from V5, unchanged)
    parser.add_argument('--ict_alpha', type=float, default=1.0,
                        help='Beta distribution param for MixUp lambda')
    parser.add_argument('--ict_weight', type=float, default=1.0,
                        help='Weight of ICT loss relative to IAC loss')

    # IAC (new in V6)
    parser.add_argument('--iac_alpha', type=float, default=2.0,
                        help='Dark-boost strength. w_dark ∈ [1, 1+alpha]')
    parser.add_argument('--iac_beta', type=float, default=2.0,
                        help='Entropy gate steepness')
    parser.add_argument('--iac_rampup', type=float, default=30.0,
                        help='Linear warmup epochs for IAC strength 0→1')

    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])

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
    print('--- Arguments ---')
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
