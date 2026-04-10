import argparse

import os

from sympy import Idx
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt



matplotlib.use('agg')
import yaml



from Utils.thresh_helper import ThreshController
from einops import rearrange
import random



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
from Models.DeepLabV3Plus_Corr.deeplabv3plus import DeepLabV3Plus
from Utils.utils import DotDict, fix_all_seed


##############################################

#            Utils

##############################################



def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    """Calculate the consistency weight for the current epoch."""
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def save_checkpoint(model, optimizer, path, epoch, metrics):
    """Save full training checkpoint with model, optimizer and metrics."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, str(path))


def load_model_state(model, path):
    """Load model from either full checkpoint or plain state_dict."""
    checkpoint = torch.load(str(path), map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)



def main(config):
    """
    Main training function.
    
    Args:
        config: Configuration object containing training parameters
    """
    # Setup datasets and dataloaders
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

    # Initialize model
    model = DeepLabV3Plus(config)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model = model.cuda()
    
    criterion_l = GeneralizedDiceFocalLoss(
        include_background=True,
        to_onehot_y=False,
        softmax=True,
        reduction='mean'
    ).cuda()

    criterion_u = nn.CrossEntropyLoss(reduction='none')
    criterion_kl = nn.KLDivLoss(reduction='none')
    
    train_val(config, model, train_loader, val_loader, criterion_l, criterion_u, criterion_kl,
              best_model_dir, best_optim_dir, file_log)
    # test(config, model, best_model_dir, test_loader, criterion_l)

    


def train_val(config, model, train_loader, val_loader, criterion_l, criterion_u, criterion_kl,
              best_model_path, best_optim_path, file_log):
    """
    Training and validation function.
    """
    # Setup optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs, eta_min=1e-6)

    # Initialize MONAI metrics with robust class fallback.
    num_classes = 3
    if hasattr(config, 'model') and getattr(config.model, 'num_classes', None) is not None:
        num_classes = config.model.num_classes
    train_dice = DiceMetric(include_background=True, 
                           num_classes=num_classes,
                           reduction="mean")

    # Update threshold controller for multiclass
    thresh_controller = ThreshController(nclass=num_classes, momentum=0.999, thresh_init=0.85)

    # Training loop
    max_dice_score = -float('inf')
    best_epoch = 0
    no_improve_epochs = 0
    early_stop_patience = config.train.early_stop_patience if getattr(config.train, 'early_stop_patience', None) is not None else 20
    early_stop_min_epochs = config.train.early_stop_min_epochs if getattr(config.train, 'early_stop_min_epochs', None) is not None else 30

    for epoch in range(config.train.num_epochs):
        start = time.time()
        model.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_loss_kl = 0.0
        total_loss_corr_ce, total_loss_corr_u = 0.0, 0.0
        total_mask_ratio = 0.0
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'], train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        train_dice.reset()



        for i, (lb_batch, u_batch, u_batch_mix) in enumerate(train_loop):
            

            img_x, mask_x = lb_batch['image'], lb_batch['label']

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = u_batch['img_w'].cuda()
            img_u_s1 = u_batch['img_s'].cuda()
            img_u_w_mix = u_batch_mix['img_w'].cuda()
            img_u_s1_mix = u_batch_mix['img_s'].cuda()

            # Build default masks for current unlabeled dataset format.
            ignore_mask = torch.zeros(img_u_w.shape[0], img_u_w.shape[2], img_u_w.shape[3], device=img_u_w.device).long()
            ignore_mask_mix = torch.zeros_like(ignore_mask)
            cutmix_box1 = torch.zeros_like(ignore_mask).bool()
            b, c, h, w = img_x.shape

            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix['out'].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)

            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            preds_corr_map = res_w['corr_map'].detach()
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s['out']
            pred_u_s1_corr = res_s['corr_out']

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.detach().argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            cutmix_box1_map = (cutmix_box1 == 1)

            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            mask_u_w_cutmixed1_copy = mask_u_w_cutmixed1.clone()
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]
            cutmix_box1_sample = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            
            if ignore_mask_cutmixed1.dim() == 4:
                ignore_mask_cutmixed1 = ignore_mask_cutmixed1[..., 0]  # Giữ đúng shape [N, H, W]

            ignore_mask_cutmixed1_sample = rearrange(ignore_mask_cutmixed1, 'n h w -> n 1 h w')

            # ignore_mask_cutmixed1_sample = rearrange((ignore_mask_cutmixed1 != 255), 'n h w -> n 1 h w')
            corr_map_u_w_cutmixed1 = (corr_map_u_w_cutmixed1 * ~cutmix_box1_sample * ignore_mask_cutmixed1_sample).bool()

            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()

            conf_fliter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255))
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w.clone()
            conf_fliter_u_w_sample = rearrange(conf_fliter_u_w_without_cutmix, 'n h w -> n 1 h w')

            segments = (corr_map_u_w_cutmixed1 * conf_fliter_u_w_sample).bool()

            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):

                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    high_conf_ratio = torch.sum(segment)/torch.sum(segment_ori)
                    if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue
                    unique_cls, count = torch.unique(mask_u_w_cutmixed1[img_idx][segment==1], return_counts=True)

                    if torch.max(count) / torch.sum(count) > thresh_global:
                        top_class = unique_cls[torch.argmax(count)]
                        mask_u_w_cutmixed1[img_idx][segment_ori==1] = top_class
                        conf_fliter_u_w_without_cutmix[img_idx] = conf_fliter_u_w_without_cutmix[img_idx] | segment_ori
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w_without_cutmix | conf_fliter_u_w

            
            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w_without_cutmix
            loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = loss_u_corr_s1 * conf_fliter_u_w_without_cutmix
            loss_u_corr_s1 = torch.sum(loss_u_corr_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_corr_s = loss_u_corr_s1

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            # Ensure ignore_mask has the correct shape by resizing if needed
            if ignore_mask.shape != conf_u_w.shape:
                ignore_mask = F.interpolate(ignore_mask.unsqueeze(1).float(), 
                                         size=conf_u_w.shape[-2:], 
                                         mode='nearest').squeeze(1)
            
            # Now both tensors should have matching dimensions
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = torch.sum(loss_u_corr_w) / torch.sum(ignore_mask != 255).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)

            loss_u_kl_sa2wa = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa, dim=1) * conf_fliter_u_w
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_kl = loss_u_kl_sa2wa

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            loss = ( 0.5 * loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Calculate metrics
            with torch.no_grad():
                # Get predictions for labeled data only
                pred_x_classes = pred_x.argmax(dim=1)  # [B, H, W]
                
                # Convert predictions to one-hot format [B, C, H, W]
                output_x_onehot = torch.zeros_like(pred_x)
                output_x_onehot.scatter_(1, pred_x_classes.unsqueeze(1), 1)

                # Convert ground truth to one-hot format [B, C, H, W]
                num_classes = pred_x.size(1)  # Get number of classes from predictions
                
                # Ensure mask_x has the right dimensions before creating one-hot
                if len(mask_x.shape) == 4:  # If mask_x is already [B, C, H, W]
                    mask_x_index = mask_x.argmax(dim=1).long()  # Convert to [B, H, W]
                else:  # If mask_x is [B, H, W]
                    mask_x_index = mask_x.long()
                
                mask_x_onehot = torch.zeros((mask_x.size(0), num_classes, *mask_x.shape[-2:]), 
                                         dtype=torch.float32, 
                                         device=mask_x.device)
                mask_x_onehot.scatter_(1, mask_x_index.unsqueeze(1), 1)

                # Update MONAI metrics - only for labeled data
                train_dice(y_pred=output_x_onehot, y=mask_x_onehot)

                # Update loss metric
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * num_lb) / (num_train + num_lb)
                num_train += num_lb

            # Update progress bar with current batch metrics
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}"
            })
            
            
            if config.debug:
                break
        
        # Get final training metrics for the epoch
        train_metrics['dice'] = train_dice.aggregate().item()

        
        # Reset metrics for next epoch
        train_dice.reset()

        
        # Log training metrics
        log_message = (f'\nEpoch {epoch}, Total train steps {i} || '
                      f'Loss: {train_metrics["loss"]:.5f}, '
                      f'Dice: {train_metrics["dice"]:.4f}, '
                        )
        print(log_message)
        file_log.write(log_message + '\n')
        file_log.flush()
        
        # Validation phase
        val_metrics = validate_model(model, val_loader)
        val_message = (f'Validation | Loss: {val_metrics["loss"]:.5f}, '
                       f'Dice: {val_metrics["dice"]:.4f}, '
                       f'IoU: {val_metrics["iou"]:.4f}, '
                       f'HD: {val_metrics["hd"]:.4f}')
        print(val_message)
        file_log.write(val_message + '\n')
        file_log.flush()
        
        # Save best model based on Dice score
        if val_metrics['dice'] > max_dice_score:
            max_dice_score = val_metrics['dice']
            best_epoch = epoch
            save_checkpoint(model, optimizer, best_model_path, epoch, val_metrics)
            torch.save(optimizer.state_dict(), best_optim_path)
            
            message = (f'New best epoch {epoch}! '
                      f'Loss: {val_metrics["loss"]:.5f}, '
                      f'Dice: {val_metrics["dice"]:.4f}, '
                      f'IoU: {val_metrics["iou"]:.4f}, '
                      f'HD: {val_metrics["hd"]:.4f}')
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch >= early_stop_min_epochs and no_improve_epochs > early_stop_patience:
            early_message = (f'Early stopping at epoch {epoch} after '
                             f'{no_improve_epochs} epochs without improvement.')
            print(early_message)
            file_log.write(early_message + '\n')
            file_log.flush()
            break
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch time
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s\n')
        print('='*80)
        if config.debug:
            break
    
    print(f'Training completed. Best epoch: {best_epoch}')
    return 



def validate_model(model, val_loader):
    """
    Validate model using MONAI metrics.
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=True, num_classes=3, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output_dict = model(img)
            # Get the main prediction output
            output = output_dict['out']

            # Compute CE loss with class-index labels.
            if label.ndim == 4 and label.shape[1] > 1:
                label_idx = label.argmax(dim=1).long()
                labels_onehot = label
            elif label.ndim == 4 and label.shape[1] == 1:
                label_idx = label.squeeze(1).long()
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label_idx.unsqueeze(1), 1)
            else:
                label_idx = label.long()
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label_idx.unsqueeze(1), 1)

            batch_loss = F.cross_entropy(output, label_idx)
            metrics['loss'] = (metrics['loss'] * num_val + batch_loss.item() * batch_len) / (num_val + batch_len)
            num_val += batch_len
            
            # Convert predictions to one-hot format
            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)
                
            # Compute metrics
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)
            
            val_loop.set_postfix({
                'Dice': f"{dice_metric.aggregate().item():.4f}"
            })
    
    # Aggregate metrics
    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()

    # Reset metrics for next validation
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
    return metrics

def test(config, model, model_dir, test_loader):
    """
    Test the model on the test set.
    
    Args:
        config: Test configuration
        model: Model to test
        model_dir: Path to saved model weights
        test_loader: Test data loader
    """
    load_model_state(model, model_dir)
    metrics = validate_model(model, test_loader)
    
    # Save and print results
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
    parser = argparse.ArgumentParser(description='Train with CorrMatch')
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
    
    # Train each fold
    for fold in [1, 2, 3, 4, 5]:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Setup directories
        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f'{exp_dir}/best.pth'
        best_optim_dir = f'{exp_dir}/best_optim.pth'
        test_results_dir = f'{exp_dir}/test_results.txt'
        
        # Save config
        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
        
        # Train fold
        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            main(config)
        
        torch.cuda.empty_cache()