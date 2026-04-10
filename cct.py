'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
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

torch.cuda.empty_cache()

def main(config):
    """
    Main training function.
    """
    # Setup datasets and dataloaders
    dataset = get_dataset(
        config, 
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment4
    )
      
    l_train_loader = torch.utils.data.DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    u_train_loader = torch.utils.data.DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )
    
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Initialize model with CCT
    model = deeplabv3plus_resnet101_cct(num_classes=3, output_stride=8, pretrained_backbone=True)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model = model.cuda()

    # Setup loss function
    criterion = DiceCELoss(
        include_background=True,
        to_onehot_y=False,
        softmax=True,
        reduction='mean'
    ).cuda()
    criterion = [criterion]

    # Train and test
    train_val(config, model, train_loader, val_loader, criterion)
    test(config, model, best_model_dir, test_loader, criterion)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def train_val(config, model, train_loader, val_loader, criterion):
    """
    Training and validation function for CCT.
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs)

    # Initialize MONAI metrics
    train_dice = DiceMetric(include_background=True, reduction="mean")
    max_dice = -float('inf')
    best_epoch = 0
    iter_num = 0
    
    # Early stopping configuration
    early_stop_patience = config.train.early_stop_patience if getattr(config.train, 'early_stop_patience', None) is not None else 10
    early_stop_min_epochs = config.train.early_stop_min_epochs if getattr(config.train, 'early_stop_min_epochs', None) is not None else 30
    no_improve_epochs = 0
    
    torch.save(model.state_dict(), best_model_dir)
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        
        train_dice.reset()
        
        for idx, (batch, batch_w_s) in enumerate(train_loop):
            # Get batch data
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            weak_batch = batch_w_s['img_w'].cuda().float()
            
            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]
            
            # Forward pass with CCT for labeled and unlabeled data
            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(img)  # labeled data
            u_outputs, u_outputs_aux1, u_outputs_aux2, u_outputs_aux3 = model(weak_batch)  # unlabeled data
            
            # Apply softmax
            outputs_soft = F.softmax(outputs, dim=1)
            outputs_aux1_soft = F.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = F.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = F.softmax(outputs_aux3, dim=1)

            u_outputs_soft = F.softmax(u_outputs, dim=1)
            u_outputs_aux1_soft = F.softmax(u_outputs_aux1, dim=1)
            u_outputs_aux2_soft = F.softmax(u_outputs_aux2, dim=1)
            u_outputs_aux3_soft = F.softmax(u_outputs_aux3, dim=1)
            
            # Calculate supervised losses for labeled data
            loss_ce = criterion[0](outputs, label)
            loss_ce_aux1 = criterion[0](outputs_aux1, label)
            loss_ce_aux2 = criterion[0](outputs_aux2, label)
            loss_ce_aux3 = criterion[0](outputs_aux3, label)
            
            # Combine supervised losses
            supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4
            
            # Calculate consistency loss for unlabeled data
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss_aux1 = torch.mean(
                (u_outputs_soft - u_outputs_aux1_soft) ** 2)
            consistency_loss_aux2 = torch.mean(
                (u_outputs_soft - u_outputs_aux2_soft) ** 2)
            consistency_loss_aux3 = torch.mean(
                (u_outputs_soft - u_outputs_aux3_soft) ** 2)

            consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3

            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check for NaN values
            if torch.isnan(consistency_loss):
                consistency_loss = torch.tensor(0.0).cuda()
            if torch.isnan(supervised_loss):
                supervised_loss = torch.tensor(0.0).cuda()

            # Total loss
            loss = supervised_loss + consistency_weight * consistency_loss
            
            # Log losses
            # print(f"Supervised Loss: {supervised_loss.item():.4f}")
            # print(f"Consistency Loss: {consistency_loss.item():.4f}")
            # print(f"Consistency Weight: {consistency_weight:.4f}")
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                output = torch.softmax(outputs, dim=1)
                output_onehot = torch.zeros_like(output)
                output_onehot.scatter_(1, output.argmax(dim=1, keepdim=True), 1)
                
                train_dice(y_pred=output_onehot, y=label)
                
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * sup_batch_len) / (num_train + sup_batch_len)
                num_train += sup_batch_len
            
            # Log training progress
            train_loop.set_postfix({
                'Loss': f"{loss.item() if not torch.isnan(loss) else 'NaN'}",
                'Sup Loss': f"{supervised_loss.item() if not torch.isnan(supervised_loss) else 'NaN'}",
                'Unsup Loss': f"{consistency_loss.item() if not torch.isnan(consistency_loss) else 'NaN'}",
                'Dice': f"{train_dice.aggregate().item():.4f}"
            })
            
            file_log.write(f'Epoch {epoch}, iter {iter_num}, Loss: {loss.item():.4f}, '
                          f'Sup Loss: {supervised_loss.item():.4f}, '
                          f'Unsup Loss: {consistency_loss.item():.4f}\n')
            file_log.flush()
            
            iter_num += 1
            if config.debug: break
        
        # Get final training metrics
        train_metrics['dice'] = train_dice.aggregate().item()
        
        # Validation phase
        val_metrics = validate_model(model, val_loader, criterion)
        
        # Save best model
        if val_metrics['dice'] > max_dice:
            max_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            no_improve_epochs = 0
            
            message = f'New best epoch {epoch}! Dice: {val_metrics["dice"]:.4f}'
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        else:
            no_improve_epochs += 1
        
        # Early stopping check
        if epoch >= early_stop_min_epochs and no_improve_epochs > early_stop_patience:
            early_message = (f'Early stopping at epoch {epoch} after {no_improve_epochs} epochs without improvement.')
            print(early_message)
            file_log.write(early_message + '\n')
            file_log.flush()
            break
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch time
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        
        if config.debug: break
    
    print(f'Training completed. Best epoch: {best_epoch}')
    return model
def validate_model(model, val_loader, criterion):
    """
    Validate a single model using MONAI metrics.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function(s)
    
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output = torch.softmax(model(img), dim=1)
            loss = criterion[0](output, label)
            
            # Convert predictions to one-hot format
            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)
            
            # Convert labels to one-hot format if needed
            if len(label.shape) == 4:  # If already one-hot
                labels_onehot = label
            else:  # If not one-hot
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)
            
            # Compute metrics
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)
            
            # Update loss
            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * batch_len) / (num_val + batch_len)
            num_val += batch_len
            
            val_loop.set_postfix({
                'Loss': f"{loss.item():.4f}"
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

def test(config, model, model_dir, test_loader, criterion):
    """
    Test the model on the test set.
    """
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)
    
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

if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str,default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/cf_fugc.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=75.0, help='consistency_rampup')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    folds_to_train = [1,2,3,4,5]
    
    for fold in folds_to_train:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        exp_dir = '{}/{}/fold{}'.format(config.data.save_folder, args.exp, fold)
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = '{}/best.pth'.format(exp_dir)
        test_results_dir = '{}/test_results.txt'.format(exp_dir)

        if config.debug == False:
            yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
            
        file_log = open('{}/log.txt'.format(exp_dir), 'w')
        
        main(config)
        
        file_log.close()
        torch.cuda.empty_cache()
