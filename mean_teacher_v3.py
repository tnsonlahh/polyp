"""
Mean Teacher V3: UniMatch-Style Dual-Stream Consistency
=======================================================

Research basis & motivation:
  All previous experiments showed that complex SSL methods (BCP, CorrMatch, CCT,
  CutMix, feature noise V2) perform EQUAL OR WORSE than vanilla Mean Teacher on
  this polyp dataset. The bottleneck is NOT the semi-supervised framework — it's
  the diversity of perturbation signals.

  Key evidence (Kvasir-SEG, DeepLabV3+ ResNet-101):
    mean_teacher_10p (MSE, mild aug):   Dice avg 0.898
    mt_enhanced_10p (CE, strong aug):   Dice avg 0.901
    mt_enhanced_v2_10p:                 Dice avg ~0.895
    BCP_20p:                            Dice avg 0.908 (vs MT_20p 0.916!)
    CorrMatch_20p:                      Dice avg 0.740 (catastrophic)

  The ONLY thing that marginally helped (+0.003): stronger augmentation.
  → V3 MAXIMIZES augmentation diversity via UniMatch's dual-stream approach.

Core idea (from UniMatch, CVPR 2023):
  Instead of 1 consistency stream (weak→strong), use 3 complementary streams:
    Stream 1: Weak→Strong_A  (geometric + noise perturbation)
    Stream 2: Weak→Strong_B  (color + obscuration perturbation)
    Stream 3: Feature noise   (logit-level perturbation, zero-cost)

  This TRIPLES the consistency signal without adding model complexity.

Additional components:
  - MSE soft consistency (proven best on this dataset, preserves distribution info)
  - Entropy minimization (encourages confident unlabeled predictions)
  - EMA teacher (stable pseudo-labels)

Changes from baseline mean_teacher.py:
  1. TripleAugment dataset: 1 weak + 2 DIFFERENT strong views (vs 1 weak only)
  2. Dual-stream MSE: consistency on both strong views independently
  3. Feature noise perturbation (uniform ±0.3 on logits, zero extra forward cost)
  4. Entropy minimization on unlabeled predictions
"""

import os
import time
import argparse
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
import albumentations as A
from torchvision import transforms
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

from Datasets.create_dataset import get_dataset, Dataset, norm01, process_multiclass_label
from Models.DeepLabV3Plus.modeling import deeplabv3plus_resnet101
from Utils.utils import DotDict, fix_all_seed


# =========================================================================
# New Dataset: TripleAugment — 1 weak + 2 independent strong augmentations
# =========================================================================
class TripleAugment(torch.utils.data.Dataset):
    """Unlabeled dataset returning 3 views: img_w, img_s1, img_s2.

    - img_w:  weak augmentation (resize + mild elastic + mild brightness)
    - img_s1: strong augmentation A (noise + elastic + blur + brightness)
              Same as existing StrongWeakAugment.strong_augment
    - img_s2: strong augmentation B (color space + grid distortion + cutout)
              Complementary to A — emphasizes different degradation modes

    Both strong views are applied to the weak-augmented base, preserving
    approximate geometric alignment with teacher pseudo-labels.
    """

    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super().__init__()
        self.dataset = dataset
        self.root_dir = data_path
        self.num_samples = len(self.dataset)

        w_p = 0.3
        s_p = 0.7

        # Weak augmentation (same as existing StrongWeakAugment)
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(alpha=10, sigma=2, p=w_p),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=w_p),
            A.GaussianBlur(blur_limit=(3, 5), p=w_p),
        ])

        # Strong A: geometric + noise/blur perturbation (from existing code)
        self.strong_augment_a = A.Compose([
            A.GaussNoise(std_range=(0.1, 0.5), p=s_p),
            A.ElasticTransform(alpha=40, sigma=4, p=s_p),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=s_p),
            A.GaussianBlur(blur_limit=(3, 7), p=s_p),
            A.MotionBlur(blur_limit=(3, 7), p=s_p),
        ])

        # Strong B: color space + obscuration perturbation (NEW, complementary to A)
        hole_size = img_size / 8 / img_size  # relative size for hole_height/width_range
        self.strong_augment_b = A.Compose([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                 val_shift_limit=20, p=s_p),
            A.RandomGamma(gamma_limit=(70, 150), p=s_p),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=s_p),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=s_p),
            A.CoarseDropout(num_holes_range=(2, 6),
                            hole_height_range=(hole_size, hole_size * 2),
                            hole_width_range=(hole_size, hole_size * 2),
                            fill=0, p=0.5),
        ])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        img_data = np.load(img_path)

        # Apply weak augmentation (base for all views)
        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']

        # Apply two independent strong augmentations to the weak base
        img_s1 = self.strong_augment_a(image=img_w.astype('uint8'))['image']
        img_s2 = self.strong_augment_b(image=img_w.astype('uint8'))['image']

        # Normalize and convert
        img_w = torch.from_numpy(norm01(img_w)).float().permute(2, 0, 1)
        img_s1 = torch.from_numpy(norm01(img_s1)).float().permute(2, 0, 1)
        img_s2 = torch.from_numpy(norm01(img_s2)).float().permute(2, 0, 1)

        return {
            'id': index,
            'img_w': img_w,
            'img_s1': img_s1,
            'img_s2': img_s2,
        }

    def __len__(self):
        return self.num_samples


# =========================================================================
# Model helpers
# =========================================================================
def create_model(ema=False):
    model = deeplabv3plus_resnet101(num_classes=3, output_stride=8,
                                    pretrained_backbone=True)
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


# =========================================================================
# Main training
# =========================================================================
def main(config):
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.1),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=TripleAugment,         # ← NEW: 3-view augmentation
    )

    l_loader = DataLoader(dataset['lb_dataset'],
                          batch_size=config.train.l_batchsize,
                          shuffle=True, num_workers=config.train.num_workers,
                          pin_memory=True, drop_last=True)
    u_loader = DataLoader(dataset['ulb_dataset'],
                          batch_size=config.train.u_batchsize,
                          shuffle=True, num_workers=config.train.num_workers,
                          pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset['val_dataset'],
                            batch_size=config.test.batch_size,
                            shuffle=False, num_workers=config.test.num_workers,
                            pin_memory=True)

    print(f"Labeled: {len(dataset['lb_dataset'])}, "
          f"Unlabeled: {len(dataset['ulb_dataset'])}, "
          f"Val: {len(dataset['val_dataset'])}")

    model = create_model().cuda()
    ema_model = create_model(ema=True).cuda()

    total_p = sum(p.numel() for p in model.parameters())
    print(f'{total_p / 1e6:.2f}M parameters')

    criterion_sup = GeneralizedDiceFocalLoss(
        softmax=True, to_onehot_y=False, include_background=True
    ).cuda()
    criterion_mse = nn.MSELoss().cuda()

    best_model = train_val(config, model, ema_model,
                           {'l_loader': l_loader, 'u_loader': u_loader},
                           val_loader, criterion_sup, criterion_mse)
    test(config, best_model, best_model_dir, val_loader, criterion_sup)


def train_val(config, model, ema_model, train_loader, val_loader,
              criterion_sup, criterion_mse):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.num_epochs, eta_min=1e-6
    )

    train_dice = DiceMetric(include_background=True, reduction="mean")

    patience = getattr(config.train, 'early_stop_patience', 20)
    min_epochs = getattr(config.train, 'early_stop_min_epochs', 30)
    no_improve = 0
    max_dice = -float('inf')
    best_epoch = 0
    global_step = 0

    for epoch in range(config.train.num_epochs):
        start = time.time()
        model.train()
        ema_model.train()
        train_dice.reset()
        train_loss_sum = 0
        num_train = 0

        source = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        loop = tqdm(source, desc=f'Epoch {epoch}', leave=False)

        for batch_l, batch_u in loop:
            # ---- Labeled data ----
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()

            # ---- Unlabeled data: 3 views ----
            img_u_w = batch_u['img_w'].cuda().float()    # weak → teacher
            img_u_s1 = batch_u['img_s1'].cuda().float()  # strong A → student
            img_u_s2 = batch_u['img_s2'].cuda().float()  # strong B → student

            # ==============================================================
            # 1. Teacher: pseudo-labels from weak view (no grad)
            # ==============================================================
            with torch.no_grad():
                teacher_out = ema_model(img_u_w)
                teacher_prob = teacher_out.softmax(dim=1)  # (B,C,H,W) soft target

            # ==============================================================
            # 2. Supervised loss on labeled data
            # ==============================================================
            outputs_l = model(img_l)
            loss_sup = criterion_sup(outputs_l, label_l)

            # ==============================================================
            # 3. Dual-stream + FP + entropy (gradient accumulation to save GPU memory)
            #    Process each stream separately, accumulate gradients.
            #    Peak memory = max(B_labeled, B_unlabeled), not sum.
            # ==============================================================
            cw = get_current_consistency_weight(epoch)

            # Backward supervised loss first (frees labeled activations)
            loss_sup.backward()

            # Stream 1: Strong A + feature perturbation + entropy
            out_s1 = model(img_u_s1)
            loss_s1 = criterion_mse(out_s1.softmax(dim=1), teacher_prob)

            # Feature noise perturbation (zero-cost: just noise on logits)
            noise = torch.empty_like(out_s1).uniform_(
                -args.fp_noise, args.fp_noise
            )
            out_fp = out_s1 + noise
            loss_fp = criterion_mse(out_fp.softmax(dim=1), teacher_prob)

            # Entropy minimization
            prob_s1 = out_s1.softmax(dim=1)
            loss_ent = -(prob_s1 * torch.log(prob_s1 + 1e-8)).sum(dim=1).mean()

            loss_stream1 = cw * (loss_s1 + args.fp_weight * loss_fp) + args.ent_weight * loss_ent
            loss_stream1.backward()  # accumulates on top of sup grads

            # Stream 2: Strong B
            out_s2 = model(img_u_s2)
            loss_s2 = criterion_mse(out_s2.softmax(dim=1), teacher_prob)

            loss_stream2 = cw * loss_s2
            loss_stream2.backward()  # accumulates on top of stream1 grads

            optimizer.step()
            optimizer.zero_grad()

            # Total loss (for logging only)
            loss = loss_sup.item() + loss_stream1.item() + loss_stream2.item()

            global_step += 1
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

            # ---- Metrics (on labeled) ----
            with torch.no_grad():
                out_oh = torch.zeros_like(outputs_l)
                out_oh.scatter_(1, outputs_l.argmax(1, keepdim=True), 1)
                train_dice(y_pred=out_oh, y=label_l)
                train_loss_sum += loss * img_l.shape[0]
                num_train += img_l.shape[0]

            loop.set_postfix({
                'Loss': f'{loss:.4f}',
                'Dice': f'{train_dice.aggregate().item():.4f}',
                'CW': f'{cw:.3f}',
            })

            if config.debug:
                break

        train_d = train_dice.aggregate().item()

        # ---- Validation ----
        val = validate_model(ema_model, val_loader, criterion_sup)

        if val['dice'] > max_dice:
            max_dice = val['dice']
            best_epoch = epoch
            torch.save(ema_model.state_dict(), best_model_dir)
            no_improve = 0
            msg = (f"New best epoch {epoch}! "
                   f"Dice: {val['dice']:.4f}, IoU: {val['iou']:.4f}, "
                   f"HD: {val['hd']:.4f}")
            print(msg)
            file_log.write(msg + '\n')
            file_log.flush()
        else:
            no_improve += 1

        if epoch >= min_epochs and no_improve > patience:
            msg = f"Early stopping at epoch {epoch}"
            print(msg)
            file_log.write(msg + '\n')
            file_log.flush()
            break

        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch} | Train Dice: {train_d:.4f} | "
              f"Val Dice: {val['dice']:.4f} | IoU: {val['iou']:.4f} | "
              f"HD: {val['hd']:.4f} | {elapsed // 60:.0f}m{elapsed % 60:.0f}s")
        print('=' * 80)

        if config.debug:
            break

    print(f"Done. Best epoch {best_epoch}, Dice: {max_dice:.4f}")
    return ema_model


# =========================================================================
# Validation & Test (standard — same as baseline)
# =========================================================================
def validate_model(model, val_loader, criterion):
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0

    dice_m = DiceMetric(include_background=True, reduction="mean")
    iou_m = MeanIoU(include_background=True, reduction="mean")
    hd_m = HausdorffDistanceMetric(include_background=True, percentile=95.0)

    for batch in tqdm(val_loader, desc='Val', leave=False):
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)

            pred_oh = torch.zeros_like(output)
            pred_oh.scatter_(1, output.argmax(1, keepdim=True), 1)

            label_oh = label if label.ndim == 4 else torch.zeros_like(output).scatter_(
                1, label.unsqueeze(1).long(), 1)

            dice_m(y_pred=pred_oh, y=label_oh)
            iou_m(y_pred=pred_oh, y=label_oh)
            hd_m(y_pred=pred_oh, y=label_oh)

            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * img.shape[0]) / (num_val + img.shape[0])
            num_val += img.shape[0]

    metrics['dice'] = dice_m.aggregate().item()
    metrics['iou'] = iou_m.aggregate().item()
    metrics['hd'] = hd_m.aggregate().item()
    dice_m.reset(); iou_m.reset(); hd_m.reset()
    return metrics


def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    m = validate_model(model, test_loader, criterion)

    txt = (f"Test Results:\n"
           f"Loss: {m['loss']:.4f}\n"
           f"Dice: {m['dice']:.4f}\n"
           f"IoU: {m['iou']:.4f}\n"
           f"HD: {m['hd']:.4f}")

    with open(test_results_dir, 'w') as f:
        f.write(txt)

    print('=' * 80)
    print(txt)
    print('=' * 80)
    file_log.write('\n' + '=' * 80 + '\n' + txt + '\n' + '=' * 80 + '\n')
    file_log.flush()


# =========================================================================
# Entry point
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mean Teacher V3 — Dual-Stream')
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
    # V3-specific
    parser.add_argument('--fp_noise', type=float, default=0.3,
                        help='Uniform noise range for feature perturbation')
    parser.add_argument('--fp_weight', type=float, default=0.5,
                        help='Weight of feature perturbation loss')
    parser.add_argument('--ent_weight', type=float, default=0.01,
                        help='Weight of entropy minimization loss')

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
            main(config)

        torch.cuda.empty_cache()
