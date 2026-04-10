"""
Mean Teacher Enhanced V2
========================
Builds on the working baseline mean_teacher_20p (Dice 0.916 avg) with 4 targeted
improvements, each independently proven in top-venue papers.

Baseline analysis:
  - mean_teacher_20p (MSE, weak-only):  Dice avg 0.916 (best: 0.929)
  - mt_enhanced (CE+strong aug):        Dice avg 0.901 (best: 0.914)
  
Key insight: On this small polyp dataset (200 labeled), MSE soft-consistency
works better than hard CE pseudo-labels. But strong augmentation adds value.
→ V2 combines BOTH: MSE on all pixels + CE on confident pixels only.

4 Changes from baseline:
  1. Weak-to-strong consistency (FixMatch/UniMatch paradigm)
     Teacher sees weak aug → student must match on strong aug
  2. Dual consistency: MSE (all pixels) + CE (confident pixels only)
     Both losses catch different failure modes
  3. CutMix on unlabeled strong views (within-batch, no extra loader)
     Forces model to handle novel context boundaries (CorrMatch, CVPR'24)
  4. Feature noise perturbation stream (UniMatch/CCT)
     Uniform noise on logits → extra consistency signal, safe for any #classes
"""

import os
import time
import argparse
from datetime import datetime
import random as pyrandom

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
from Datasets.transform import obtain_cutmix_box
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed


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


def main(config):
    # CHANGE 1: StrongWeakAugment for weak-to-strong paradigm
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment
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

    # CHANGE 2: Dual consistency losses
    criterion_mse = nn.MSELoss().cuda()           # soft targets, all pixels
    criterion_ce = nn.CrossEntropyLoss(reduction='none').cuda()  # hard pseudo-labels, confident pixels

    best_model = train_val(config, model, ema_model, train_loader, val_loader,
                           criterion_sup, criterion_mse, criterion_ce)
    test(config, best_model, best_model_dir, test_loader, criterion_sup)


def train_val(config, model, ema_model, train_loader, val_loader,
              criterion_sup, criterion_mse, criterion_ce):
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

            img_u_w = batch_u['img_w'].cuda().float()  # weak view → teacher
            img_u_s = batch_u['img_s'].cuda().float()  # strong view → student

            b, c, h, w = img_u_s.shape

            # =============================================
            # Teacher: pseudo-labels from weak view
            # =============================================
            with torch.no_grad():
                teacher_out = ema_model(img_u_w)
                teacher_prob = teacher_out.softmax(dim=1)         # (B,C,H,W)
                teacher_conf = teacher_prob.max(dim=1)[0]         # (B,H,W)
                pseudo_labels = teacher_out.argmax(dim=1)          # (B,H,W)

            # =============================================
            # CHANGE 3: CutMix within batch
            # Shuffle batch and mix regions of strong images
            # =============================================
            indices = torch.randperm(b, device=img_u_s.device)
            cutmix_box = obtain_cutmix_box(h, p=0.5).cuda()      # (H,W)
            cutmix_mask = (cutmix_box == 1)                        # bool (H,W)

            # Mix strong images
            img_u_s_mixed = img_u_s.clone()
            if cutmix_mask.any():
                img_u_s_mixed[:, :, cutmix_mask] = img_u_s[indices][:, :, cutmix_mask]

            # Mix pseudo-labels and confidence accordingly
            pseudo_mixed = pseudo_labels.clone()
            conf_mixed = teacher_conf.clone()
            if cutmix_mask.any():
                pseudo_mixed[:, cutmix_mask] = pseudo_labels[indices][:, cutmix_mask]
                conf_mixed[:, cutmix_mask] = teacher_conf[indices][:, cutmix_mask]

            conf_filter = (conf_mixed >= args.conf_thresh).float()

            # =============================================
            # Student forward passes
            # =============================================
            # Supervised
            outputs_l = model(img_l)
            sup_loss = criterion_sup(outputs_l, label_l)

            # Stream 1: Strong aug + CutMix → student prediction
            outputs_u_s = model(img_u_s_mixed)

            # CE loss on confident pseudo-label pixels (hard targets)
            loss_ce = criterion_ce(outputs_u_s, pseudo_mixed)
            loss_ce = (loss_ce * conf_filter).sum() / (conf_filter.sum() + 1e-6)

            # MSE loss on all pixels (soft targets) — preserves distribution info
            # Use teacher prob with matching CutMix regions
            teacher_prob_mixed = teacher_prob.clone()
            if cutmix_mask.any():
                teacher_prob_mixed[:, :, cutmix_mask] = teacher_prob[indices][:, :, cutmix_mask]
            loss_mse = criterion_mse(outputs_u_s.softmax(dim=1), teacher_prob_mixed)

            # =============================================
            # CHANGE 4: Feature noise perturbation stream
            # Add uniform noise to student logits on weak view
            # Like FeatureNoise in CCT (CVPR'20) / UniMatch
            # =============================================
            outputs_u_w = model(img_u_w)
            noise = torch.empty_like(outputs_u_w).uniform_(
                -args.fp_noise, args.fp_noise
            )
            outputs_u_w_fp = outputs_u_w + noise

            # FP stream: CE on confident + MSE on all (using original pseudo-labels)
            conf_filter_w = (teacher_conf >= args.conf_thresh).float()
            loss_fp_ce = criterion_ce(outputs_u_w_fp, pseudo_labels)
            loss_fp_ce = (loss_fp_ce * conf_filter_w).sum() / (conf_filter_w.sum() + 1e-6)
            loss_fp_mse = criterion_mse(outputs_u_w_fp.softmax(dim=1), teacher_prob)

            loss_fp = loss_fp_ce + args.mse_weight * loss_fp_mse

            # =============================================
            # Total loss
            # =============================================
            consistency_weight = get_current_consistency_weight(epoch)
            loss_unsup = loss_ce + args.mse_weight * loss_mse + args.fp_weight * loss_fp

            loss = sup_loss + consistency_weight * loss_unsup

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

            mask_ratio = conf_filter.mean().item()
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}",
                'Mask': f"{mask_ratio:.2f}"
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
              f'{time_elapsed//60:.0f}m{time_elapsed%60:.0f}s')
        print('=' * 80)

        if config.debug:
            break

    print(f'Training completed. Best epoch: {best_epoch}, Dice: {max_dice:.4f}')
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

    print('=' * 80)
    print(results_str)
    print('=' * 80)
    file_log.write('\n' + '=' * 80 + '\n')
    file_log.write(results_str + '\n')
    file_log.write('=' * 80 + '\n')
    file_log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mean Teacher Enhanced V2')
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
    # New hyperparameters (with conservative defaults)
    parser.add_argument('--conf_thresh', type=float, default=0.95,
                        help='Confidence threshold for pseudo-label CE')
    parser.add_argument('--mse_weight', type=float, default=1.0,
                        help='Weight of MSE soft consistency relative to CE')
    parser.add_argument('--fp_noise', type=float, default=0.3,
                        help='Uniform noise range for feature perturbation')
    parser.add_argument('--fp_weight', type=float, default=0.5,
                        help='Weight of feature perturbation stream')

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
        print(f"{arg:<25}: {getattr(args, arg)}")

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
