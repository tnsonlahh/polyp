"""
Dark-Region Aware Mean Teacher V2 (DRA-MT V2)
==============================================
Semi-supervised polyp segmentation with improved handling of dark corners
and deep folds, incorporating insights from recent SOTA methods:

1. UniMatch (CVPR 2023) — Dual-stream weak-to-strong consistency:
   - Image-level strong augmentation stream
   - Feature-level perturbation stream (dropout on features)
   - Both guided by teacher pseudo-labels from weak view

2. CorrMatch (CVPR 2024) — Adaptive confidence thresholding:
   - Class-adaptive momentum threshold (ThreshController)
   - Pseudo-label CE replaces MSE consistency (much more effective)
   - CutMix augmentation for better unlabeled utilization

3. Entropy-Guided Regional Focus (for dark/deep regions):
   - Teacher uncertainty (entropy) as proxy for "hard regions"
   - Dark corners have high teacher entropy → receive higher loss weight
   - Principled alternative to crude intensity-based weighting

4. Boundary-Aware Structure Loss (on labeled data):
   - Auxiliary edge supervision sharpens segmentation boundaries
   - Critical for polyps in low-contrast areas

Key difference from vanilla Mean Teacher:
  MSE consistency → Pseudo-label CE with confidence thresholding
  Single view     → Dual-stream perturbation (image + feature)
  Flat weighting  → Entropy-guided focusing on hard regions
  No mixing       → CutMix on unlabeled data
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
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

from Datasets.create_dataset import *
from Datasets.transform import obtain_cutmix_box
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed
from Utils.thresh_helper import ThreshController


# ==============================================================
# Module 1: Feature Perturbation (from UniMatch)
# ==============================================================

class FeaturePerturbation(nn.Module):
    """
    Apply dropout-based perturbation to model features to create an
    auxiliary prediction stream. The teacher sees clean features;
    the student must match even with perturbed features.
    This doubles the perturbation space beyond image-level augmentation.
    
    Reference: UniMatch (CVPR 2023) — feature-level perturbation stream
    """
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        return self.drop(x)


# ==============================================================
# Module 2: Entropy-Guided Focusing (for dark/deep regions)
# ==============================================================

def compute_entropy_weights(teacher_probs, max_boost=2.0):
    """
    Use teacher prediction entropy as a proxy for region difficulty.
    Dark corners and deep folds → teacher is uncertain → high entropy
    → increase loss weight to force student-teacher agreement there.
    
    Unlike intensity-based weighting, this is model-aware and adapts
    as the teacher improves.
    
    Args:
        teacher_probs: softmax output from teacher (B, C, H, W)
        max_boost: maximum additional weight for high-entropy regions
    Returns:
        weight_map: (B, 1, H, W) per-pixel weights in [1, 1+max_boost]
    """
    entropy = -(teacher_probs * (teacher_probs + 1e-10).log()).sum(dim=1, keepdim=True)
    max_entropy = np.log(teacher_probs.shape[1])  # log(C)
    normalized_entropy = entropy / max_entropy  # [0, 1]
    weight_map = 1.0 + max_boost * normalized_entropy
    return weight_map


# ==============================================================
# Module 3: Boundary-Aware Loss (for sharp edges)
# ==============================================================

def extract_boundary(label_onehot, kernel_size=3):
    """Extract boundaries from one-hot labels using morphological ops."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(label_onehot, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-label_onehot, kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).sum(dim=1, keepdim=True).clamp(0, 1)
    return (boundary > 0).float()


class BoundaryAwareLoss(nn.Module):
    """
    CE loss with higher weight at object boundaries.
    Polyps in dark corners have weak, blurry edges; this forces
    sharper segmentation at boundaries.
    Only applied to labeled data (reliable GT).
    """
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def forward(self, pred, label_onehot):
        boundary = extract_boundary(label_onehot)
        weight_map = 1.0 + self.boundary_weight * boundary
        target = label_onehot.argmax(dim=1)
        ce = F.cross_entropy(pred, target, reduction='none')
        return (ce * weight_map.squeeze(1)).mean()


# ==============================================================
# Model Helpers
# ==============================================================

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


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


# ==============================================================
# Main Training
# ==============================================================

def main(config, args):
    # Use StrongWeakAugment (not StrongWeakAugment4) for proper strong+weak views
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment  # Strong+Weak augmentation
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
    # Second unlabeled loader for CutMix partner
    u_train_loader_mix = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True, num_workers=config.train.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False, num_workers=config.test.num_workers, pin_memory=True
    )

    train_loader = {
        'l_loader': l_train_loader,
        'u_loader': u_train_loader,
        'u_loader_mix': u_train_loader_mix
    }
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Create student and teacher models
    model = create_model().cuda()
    ema_model = create_model(ema=True).cuda()

    # Feature perturbation module (UniMatch-style)
    feat_perturb = FeaturePerturbation(drop_rate=args.fp_drop_rate).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params / 1e6:.2f}M total params, {total_trainable / 1e6:.2f}M trainable')

    # Loss functions
    criterion_sup = GeneralizedDiceFocalLoss(
        softmax=True, to_onehot_y=False, include_background=True
    ).cuda()
    criterion_unsup = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_boundary = BoundaryAwareLoss(boundary_weight=args.boundary_weight).cuda()

    # Adaptive threshold controller (from CorrMatch)
    thresh_controller = ThreshController(nclass=3, momentum=0.999, thresh_init=args.conf_thresh)

    best_model = train_val(config, args, model, ema_model, feat_perturb,
                           train_loader, val_loader,
                           criterion_sup, criterion_unsup, criterion_boundary,
                           thresh_controller)
    test(config, best_model, best_model_dir, val_loader, criterion_sup)


def train_val(config, args, model, ema_model, feat_perturb,
              train_loader, val_loader,
              criterion_sup, criterion_unsup, criterion_boundary,
              thresh_controller):

    optimizer = optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(feat_perturb.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs, eta_min=1e-6
    )

    train_dice = DiceMetric(include_background=True, reduction="mean")
    early_stop_patience = getattr(config.train, 'early_stop_patience', 20) or 20
    early_stop_min_epochs = getattr(config.train, 'early_stop_min_epochs', 50) or 50
    no_improve_epochs = 0
    max_dice = -float('inf')
    best_epoch = 0
    global_step = 0

    for epoch in range(config.train.num_epochs):
        start = time.time()
        model.train()
        ema_model.train()
        feat_perturb.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0
        train_dice.reset()

        source = zip(
            cycle(train_loader['l_loader']),
            train_loader['u_loader'],
            train_loader['u_loader_mix']
        )
        train_loop = tqdm(source, desc=f'Epoch {epoch}', leave=False)

        for batch_idx, (batch_l, batch_u, batch_u_mix) in enumerate(train_loop):
            # === Labeled data ===
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()

            # === Unlabeled data ===
            img_u_w = batch_u['img_w'].cuda().float()      # weak augmentation
            img_u_s = batch_u['img_s'].cuda().float()      # strong augmentation
            img_u_w_mix = batch_u_mix['img_w'].cuda().float()  # CutMix partner

            b, c, h, w = img_u_w.shape

            # --------------------------------------------------
            # Step 1: Teacher generates pseudo-labels (weak view)
            # --------------------------------------------------
            with torch.no_grad():
                ema_model.eval()
                pred_u_w_teacher = ema_model(img_u_w)       # (B, C, H, W)
                pred_u_w_mix_teacher = ema_model(img_u_w_mix)

                prob_u_w = pred_u_w_teacher.softmax(dim=1)
                prob_u_w_mix = pred_u_w_mix_teacher.softmax(dim=1)

                conf_u_w = prob_u_w.max(dim=1)[0]           # (B, H, W)
                mask_u_w = pred_u_w_teacher.argmax(dim=1)    # pseudo-labels

                conf_u_w_mix = prob_u_w_mix.max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix_teacher.argmax(dim=1)

            # --------------------------------------------------
            # Step 2: CutMix — mix unlabeled samples
            # --------------------------------------------------
            cutmix_box = obtain_cutmix_box(h, p=0.5).unsqueeze(0).cuda()  # (1, H, W)
            cutmix_mask = (cutmix_box == 1)

            # Mix strong images
            img_u_s_mixed = img_u_s.clone()
            img_u_s_mixed[cutmix_mask.unsqueeze(1).expand_as(img_u_s)] = \
                batch_u_mix['img_s'].cuda().float()[cutmix_mask.unsqueeze(1).expand_as(img_u_s)]

            # Mix pseudo-labels and confidence accordingly
            mask_u_cutmixed = mask_u_w.clone()
            conf_u_cutmixed = conf_u_w.clone()
            mask_u_cutmixed[cutmix_mask.expand_as(mask_u_w)] = mask_u_w_mix[cutmix_mask.expand_as(mask_u_w)]
            conf_u_cutmixed[cutmix_mask.expand_as(conf_u_w)] = conf_u_w_mix[cutmix_mask.expand_as(conf_u_w)]

            # --------------------------------------------------
            # Step 3: Update adaptive threshold
            # --------------------------------------------------
            with torch.no_grad():
                ignore_mask = torch.zeros(b, h, w, device=img_u_w.device).long()
                thresh_controller.thresh_update(pred_u_w_teacher.detach(), ignore_mask, update_g=True)
                thresh = thresh_controller.get_thresh_global()

            conf_filter = (conf_u_cutmixed >= thresh)

            # --------------------------------------------------
            # Step 4: Student forward passes
            # --------------------------------------------------
            model.train()

            # 4a. Supervised on labeled data
            outputs_l = model(img_l)
            sup_loss = criterion_sup(outputs_l, label_l)
            boundary_loss = criterion_boundary(outputs_l, label_l)

            # 4b. Stream 1: Image-level strong augmentation (UniMatch)
            pred_u_s = model(img_u_s_mixed)
            loss_u_s = criterion_unsup(pred_u_s, mask_u_cutmixed)
            loss_u_s = (loss_u_s * conf_filter).sum() / (conf_filter.sum() + 1e-6)

            # 4c. Stream 2: Feature-level perturbation (UniMatch)
            #     Pass weak image through student, but perturb features
            #     by running another forward with dropout applied to output
            pred_u_w_student = model(img_u_w)
            # Apply feature perturbation to logits before softmax
            pred_u_w_fp = feat_perturb(pred_u_w_student)
            # Use original (non-cutmixed) pseudo-labels for FP stream
            conf_filter_w = (conf_u_w >= thresh)
            loss_u_fp = criterion_unsup(pred_u_w_fp, mask_u_w)
            loss_u_fp = (loss_u_fp * conf_filter_w).sum() / (conf_filter_w.sum() + 1e-6)

            # --------------------------------------------------
            # Step 5: Entropy-guided weighting for hard regions
            # --------------------------------------------------
            if args.entropy_weight > 0:
                with torch.no_grad():
                    entropy_w = compute_entropy_weights(prob_u_w, max_boost=args.entropy_weight)
                    entropy_w_squeezed = entropy_w.squeeze(1)  # (B, H, W)

                # Re-compute strong-view loss with entropy weighting
                loss_u_s_raw = criterion_unsup(pred_u_s, mask_u_cutmixed)
                loss_u_s_entropy = (loss_u_s_raw * conf_filter * entropy_w_squeezed)
                loss_u_s = loss_u_s_entropy.sum() / (conf_filter.sum() + 1e-6)

                # Re-compute FP loss with entropy weighting
                loss_u_fp_raw = criterion_unsup(pred_u_w_fp, mask_u_w)
                loss_u_fp_entropy = (loss_u_fp_raw * conf_filter_w * entropy_w_squeezed)
                loss_u_fp = loss_u_fp_entropy.sum() / (conf_filter_w.sum() + 1e-6)

            # --------------------------------------------------
            # Step 6: Total loss
            # --------------------------------------------------
            consistency_weight = get_current_consistency_weight(
                epoch, args.consistency, args.consistency_rampup
            )

            loss = (sup_loss
                    + args.boundary_loss_weight * boundary_loss
                    + consistency_weight * (loss_u_s + args.fp_weight * loss_u_fp))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher
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
                'Thr': f"{thresh.item():.3f}"
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
            print(f'Early stopping at epoch {epoch}.')
            file_log.write(f'Early stopping at epoch {epoch}.\n')
            file_log.flush()
            break

        scheduler.step()
        elapsed = time.time() - start
        print(f'Epoch {epoch} | Train Dice: {train_metrics["dice"]:.4f} | '
              f'Val Dice: {val_metrics["dice"]:.4f} | '
              f'Val IoU: {val_metrics["iou"]:.4f} | '
              f'HD: {val_metrics["hd"]:.4f} | '
              f'Time: {elapsed // 60:.0f}m{elapsed % 60:.0f}s')
        print('=' * 80)

        if config.debug:
            break

    print(f'Best epoch: {best_epoch}, Best Dice: {max_dice:.4f}')
    return ema_model


def validate_model(model, val_loader, criterion):
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)

    for batch in tqdm(val_loader, desc='Validation', leave=False):
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)

            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, output.argmax(dim=1, keepdim=True), 1)

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

    print('=' * 80)
    print(results_str)
    print('=' * 80)
    file_log.write('\n' + '=' * 80 + '\n' + results_str + '\n' + '=' * 80 + '\n')
    file_log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRA-MT V2: Dark-Region Aware Mean Teacher')
    parser.add_argument('--exp', type=str, default='dramt_v2')
    parser.add_argument('--config_yml', type=str, default='Configs/kvasir_seg.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='kvasir')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)

    # Mean Teacher core
    parser.add_argument('--consistency', type=float, default=1.0,
                        help='Consistency loss weight (higher than vanilla MT due to pseudo-label CE)')
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Adaptive threshold (from CorrMatch)
    parser.add_argument('--conf_thresh', type=float, default=0.85,
                        help='Initial confidence threshold for pseudo-labels')

    # Feature perturbation (from UniMatch)
    parser.add_argument('--fp_drop_rate', type=float, default=0.5,
                        help='Dropout rate for feature perturbation stream')
    parser.add_argument('--fp_weight', type=float, default=0.5,
                        help='Weight of feature perturbation loss relative to strong aug loss')

    # Entropy-guided focusing (for dark regions)
    parser.add_argument('--entropy_weight', type=float, default=1.0,
                        help='Max entropy boost for hard (dark) regions. 0 to disable')

    # Boundary loss
    parser.add_argument('--boundary_weight', type=float, default=2.0,
                        help='Extra CE weight at object boundaries')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.3,
                        help='Weight of boundary loss in total loss')

    args = parser.parse_args()

    # Load config
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
        print(f"{arg:<25}: {getattr(args, arg)}")

    store_config = config
    config = DotDict(config)

    for fold in [1, 2, 3, 4, 5]:
        print(f"\n{'=' * 80}")
        print(f"=== Training Fold {fold} ===")
        print(f"{'=' * 80}")
        config['fold'] = fold

        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f'{exp_dir}/best.pth'
        test_results_dir = f'{exp_dir}/test_results.txt'

        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))

        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            main(config, args)

        torch.cuda.empty_cache()
