"""
IDA-MT: Illumination- and Depth-Aware Mean Teacher for Semi-Supervised Polyp Segmentation
==========================================================================================

Motivation
----------
Colonoscopy images have two physics-specific biases that standard Mean Teacher ignores:
  (1) Illumination: endoscope lighting + radial vignetting → 30-60% luminance loss
      at image corners. Polyps in dark regions are systematically under-segmented
      by the teacher, yet MT treats all pixels equally in the consistency loss.
  (2) Depth: polyps far from the camera are small, low-contrast, with blurred
      boundaries. These are the highest-error cases clinically but the hardest
      for the teacher to label reliably.

We address both at once with two orthogonal, complementary mechanisms:

  C1. Illumination-Aware Consistency (IAC)
      Per-pixel consistency weight = dark-boost × entropy-gate.
      - Dark pixels get amplified consistency loss (where teacher typically fails)
      - Entropy gate prevents learning from confident-wrong garbage
      - Warmup schedule avoids confirmation bias early in training

  C2. Scale-illumination Consistency under Induced Perturbation (SCIP)
      Simulate "polyp far from camera" by downscale + darken on LABELED images.
      Two losses:
        - Supervised Dice on perturbed view (student must still segment correctly)
        - Consistency with teacher prediction on original scale (scale invariance)
      This gives labeled data a second gradient pathway targeting depth robustness.

Framework: standard Mean Teacher (single student, EMA teacher). No architectural changes.
Backbone: DeepLabV3+ ResNet101 (unchanged from baseline).

Loss:
    L = L_sup(labeled) + λ_c(t) · [L_IAC(unlabeled) + λ_s · L_SCIP(labeled)]

where λ_c(t) is sigmoid rampup (same as vanilla MT) and λ_s is a fixed weight.
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
# Augmentation — fixed, reproducible, no knob-tuning menagerie
# ============================================================

class RobustStrongWeakAugment(StrongWeakAugment):
    """
    Clean FixMatch-style weak/strong split. Strong augmentation is the
    standard photometric + geometric perturbation set. We DO NOT add
    domain-specific augmentations here — the domain-specific handling
    lives in IAC and SCIP. Keeping this clean means clean ablations.
    """
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super().__init__(dataset, img_size, use_aug, data_path)
        s_p = 0.2
        self.strong_augment = A.Compose([
            A.GaussNoise(std_range=(0.005, 0.02), p=s_p),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=s_p),
            A.GaussianBlur(blur_limit=(3, 3), p=s_p),
        ])


# ============================================================
# C1: Illumination-Aware Consistency (IAC)
# ============================================================

def compute_luminance(img: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel luminance from RGB input, smoothed with Gaussian kernel
    to capture local illumination rather than pixel-level noise.

    Args:
        img: (B, 3, H, W) RGB in [0, 1] or standard-normalized.
             We use Rec.601 luminance: L = 0.299 R + 0.587 G + 0.114 B
             which matches HSV V-channel in monotonic ordering.
    Returns:
        L: (B, 1, H, W) luminance in [0, 1] — normalized per-image
           so that we capture RELATIVE darkness within each image
           (corner dark vs center bright), not absolute darkness.
    """
    # Rec.601 luminance — works even if img is not in [0,1] since we normalize after
    L = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    # Smooth with 15x15 average pool — captures regional illumination
    L = F.avg_pool2d(L, kernel_size=15, stride=1, padding=7)

    # Per-image min-max normalize so IAC is robust to global brightness shifts
    B = L.shape[0]
    L_flat = L.view(B, -1)
    L_min = L_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    L_max = L_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    L = (L - L_min) / (L_max - L_min + 1e-6)
    return L  # (B, 1, H, W), per-image [0, 1]


def iac_weight(img: torch.Tensor, teacher_probs: torch.Tensor,
               alpha: float = 2.0, beta: float = 2.0) -> torch.Tensor:
    """
    Illumination-Aware Consistency weight map.

        w(x,y) = (1 + α·(1 - L(x,y))) · exp(-β·H[p_t(x,y)])
                 └─── dark boost ───┘   └─── entropy gate ─┘

    - dark pixels (L≈0)  →  dark_boost ≈ 1+α  (amplified consistency)
    - bright pixels (L≈1) → dark_boost ≈ 1    (standard consistency)
    - uncertain teacher   →  gate ≈ 0         (skip — avoid confirmation bias)
    - confident teacher   →  gate ≈ 1         (full weight)

    Args:
        img: (B, 3, H, W) unlabeled image (weak view)
        teacher_probs: (B, C, H, W) teacher softmax output
        alpha: dark boost strength (default 2.0 → up to 3x weight on darkest pixels)
        beta: entropy gate steepness
    Returns:
        w: (B, 1, H, W) per-pixel consistency weight
    """
    # Dark boost
    L = compute_luminance(img)                                # (B, 1, H, W)
    dark_boost = 1.0 + alpha * (1.0 - L)                      # [1, 1+α]

    # Entropy gate — low entropy = confident teacher = trustworthy
    # H = -Σ p log p, normalized to [0, 1] by dividing by log(C)
    C = teacher_probs.shape[1]
    H = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=1, keepdim=True)
    H_norm = H / np.log(C)                                    # (B, 1, H, W)
    gate = torch.exp(-beta * H_norm)                          # [exp(-β), 1]

    return dark_boost * gate


def iac_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor,
             img: torch.Tensor, alpha: float = 2.0, beta: float = 2.0,
             iac_strength: float = 1.0) -> torch.Tensor:
    """
    Illumination-Aware Consistency loss. Replaces vanilla MSE consistency.

    iac_strength ∈ [0, 1] — warmup scaler.
        0 = pure vanilla MSE (no IAC)
        1 = full IAC weighting
    Linear warmup from 0→1 over first N epochs avoids the confirmation bias
    trap where downweighting happens before teacher has stabilized.
    """
    student_probs = student_logits.softmax(dim=1)
    # MSE per pixel, summed over channels
    sq_err = (student_probs - teacher_probs).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)

    w = iac_weight(img, teacher_probs, alpha=alpha, beta=beta)  # (B,1,H,W)

    # Blend between uniform (MSE) and weighted-MSE based on warmup
    w_blend = (1.0 - iac_strength) * torch.ones_like(w) + iac_strength * w

    # Normalize so loss magnitude is comparable across warmup levels
    w_blend = w_blend / (w_blend.mean(dim=(2, 3), keepdim=True) + 1e-6)

    return (w_blend * sq_err).mean()


# ============================================================
# C2: Scale-illumination Consistency under Induced Perturbation (SCIP)
# ============================================================

def apply_scip_perturbation(img: torch.Tensor, label: torch.Tensor,
                            scale_range=(0.4, 0.7),
                            darken_range=(0.5, 0.8)):
    """
    Simulate "polyp far from camera in a dark corner":
      (1) Random downscale by factor s, then zoom back to original size
          → blurs fine detail, shrinks apparent polyp size
      (2) Random multiplicative darkening by factor γ
          → simulates reduced illumination at depth

    Label is downscaled/upscaled with NEAREST to preserve class boundaries.

    Returns:
        img_far: (B, 3, H, W) perturbed image
        label_far: (B, C, H, W) perturbed label (if provided) or None
    """
    B, _, H, W = img.shape
    s = np.random.uniform(*scale_range)
    gamma = np.random.uniform(*darken_range)

    # Downscale then upscale — creates blur + scale shift
    h_small, w_small = max(1, int(H * s)), max(1, int(W * s))
    img_down = F.interpolate(img, size=(h_small, w_small), mode='bilinear', align_corners=False)
    img_far = F.interpolate(img_down, size=(H, W), mode='bilinear', align_corners=False)

    # Darken
    img_far = img_far * gamma

    if label is not None:
        lbl_down = F.interpolate(label, size=(h_small, w_small), mode='nearest')
        label_far = F.interpolate(lbl_down, size=(H, W), mode='nearest')
    else:
        label_far = None

    return img_far, label_far


def scip_loss(model, ema_model, img_l: torch.Tensor, label_l: torch.Tensor,
              criterion_sup, criterion_mse) -> torch.Tensor:
    """
    SCIP combines two signals on the LABELED data:
      (a) L_sup_far:  supervised Dice on perturbed view — teaches the student
          to segment polyps correctly even at reduced scale/illumination
      (b) L_cons_far: consistency between student(perturbed) and teacher(original)
          — teaches scale-invariance using real labels as anchor

    Both losses operate on the SAME labeled batch to avoid extra forward passes
    on unlabeled data. This is efficient and makes SCIP contribution interpretable.
    """
    img_far, label_far = apply_scip_perturbation(img_l, label_l)

    # Teacher prediction on ORIGINAL (unperturbed) view — serves as scale-invariant target
    with torch.no_grad():
        teacher_orig = ema_model(img_l).softmax(dim=1)

    # Student on perturbed view
    student_far_logits = model(img_far)

    # (a) Supervised loss on perturbed view with perturbed label
    L_sup_far = criterion_sup(student_far_logits, label_far)

    # (b) Consistency: student(perturbed) should match teacher(original) up to scale
    #     No need to transform — both are at original resolution after interpolation
    L_cons_far = criterion_mse(student_far_logits.softmax(dim=1), teacher_orig)

    return L_sup_far + L_cons_far


# ============================================================
# Model utilities (unchanged from baseline)
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
    """For IAC warmup — linear from 0 to 1."""
    if rampup_length == 0:
        return 1.0
    return float(np.clip(current / rampup_length, 0.0, 1.0))


def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def get_iac_strength(epoch):
    """
    IAC warmup: epoch 0 → strength=0 (vanilla MSE)
                epoch ≥ iac_rampup → strength=1 (full IAC).
    Rationale: early teacher is noisy; applying aggressive dark-boost weighting
    on a noisy teacher amplifies wrong pseudo-labels. Warm up after teacher settles.
    """
    return linear_rampup(epoch, args.iac_rampup)


# ============================================================
# Main training pipeline
# ============================================================

def main(config):
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
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

        consistency_weight = get_current_consistency_weight(epoch)
        iac_strength = get_iac_strength(epoch) if args.use_iac else 0.0

        for batch_idx, (batch_l, batch_u) in enumerate(train_loop):
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()
            img_u_w = batch_u['img_w'].cuda().float()
            img_u_s = batch_u['img_s'].cuda().float()

            # --- Supervised on labeled ---
            outputs_l = model(img_l)
            sup_loss = criterion_sup(outputs_l, label_l)

            # --- Teacher prediction on unlabeled weak view ---
            with torch.no_grad():
                teacher_probs = ema_model(img_u_w).softmax(dim=1)

            # --- C1: IAC consistency on unlabeled ---
            student_s_logits = model(img_u_s)
            if args.use_iac:
                cons_loss = iac_loss(
                    student_s_logits, teacher_probs, img_u_w,
                    alpha=args.iac_alpha, beta=args.iac_beta,
                    iac_strength=iac_strength
                )
            else:
                # Vanilla MSE consistency fallback (for ablation baseline)
                cons_loss = criterion_mse(student_s_logits.softmax(dim=1), teacher_probs)

            # --- C2: SCIP on labeled ---
            if args.use_scip:
                scip_l = scip_loss(model, ema_model, img_l, label_l, criterion_sup, criterion_mse)
            else:
                scip_l = torch.tensor(0.0, device=img_l.device)

            # --- Total loss ---
            loss = sup_loss + consistency_weight * cons_loss + args.scip_weight * scip_l

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
                'Cons': f"{cons_loss.item():.3f}",
                'SCIP': f"{scip_l.item():.3f}",
                'iac_s': f"{iac_strength:.2f}",
                'Dice': f"{train_dice.aggregate().item():.3f}"
            })

            if config.debug:
                break

        train_metrics['dice'] = train_dice.aggregate().item()

        # --- Validation (teacher model — standard MT protocol) ---
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
# 
        # Early stop disabled

        scheduler.step()
        dt = time.time() - start
        print(f'Ep {epoch} | Train D {train_metrics["dice"]:.4f} | Val D {val_metrics["dice"]:.4f} '
              f'| IoU {val_metrics["iou"]:.4f} | HD {val_metrics["hd"]:.4f} | {dt//60:.0f}m{dt%60:.0f}s')
        print('='*80)

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

    dice_metric.reset(); iou_metric.reset(); hd_metric.reset()
    return metrics


def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)

    results_str = (f"Test Results:\nLoss: {metrics['loss']:.4f}\n"
                   f"Dice: {metrics['dice']:.4f}\nIoU: {metrics['iou']:.4f}\nHD: {metrics['hd']:.4f}")

    with open(test_results_dir, 'w') as f:
        f.write(results_str)

    print('='*80); print(results_str); print('='*80)
    file_log.write('\n' + '='*80 + '\n' + results_str + '\n' + '='*80 + '\n')
    file_log.flush()


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDA-MT: Illumination-Depth Aware Mean Teacher')
    parser.add_argument('--exp', type=str, default='ida_mt')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)

    # Mean Teacher core
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # ---- Ablation switches ----
    parser.add_argument('--use_iac', action='store_true',
                        help='Enable C1: Illumination-Aware Consistency')
    parser.add_argument('--use_scip', action='store_true',
                        help='Enable C2: Scale+illumination Perturbation consistency')

    # ---- IAC hyperparameters ----
    parser.add_argument('--iac_alpha', type=float, default=2.0,
                        help='Dark-boost strength. w_dark ∈ [1, 1+α]')
    parser.add_argument('--iac_beta', type=float, default=2.0,
                        help='Entropy gate steepness.')
    parser.add_argument('--iac_rampup', type=float, default=30.0,
                        help='Linear warmup epochs for IAC strength 0→1')

    # ---- SCIP hyperparameters ----
    parser.add_argument('--scip_weight', type=float, default=0.5,
                        help='Weight of SCIP loss relative to supervised loss')

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
    print(f'IAC: {"ON" if args.use_iac else "OFF"} | SCIP: {"ON" if args.use_scip else "OFF"}')

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
