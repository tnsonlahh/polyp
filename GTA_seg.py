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
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed

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
        ulb_dataset=StrongWeakAugment4
    )
      
    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Initialize models
    model_student = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True)
    model_teacher = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True)
    model_ta = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True)

    # Print model statistics
    total_params = sum(p.numel() for p in model_student.parameters())
    total_trainable_params = sum(p.numel() for p in model_student.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model_student = model_student.cuda()
    model_teacher = model_teacher.cuda()
    model_ta = model_ta.cuda()

    # Setup loss function
    criterion = GeneralizedDiceFocalLoss(
        include_background=True,
        to_onehot_y=False,
        softmax=True,
        reduction='mean'
    ).cuda()
    criterion = [criterion]
    
    global loss_weights
    loss_weights = [1.0]

    # Train and test
    model = train_val(config, model_student, model_teacher, model_ta, train_loader, val_loader, criterion)
    test(config, model, best_model_dir, test_loader, criterion)

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
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def train_val(config, model, model_teacher, model_ta, train_loader, val_loader, criterion):
    """
    Training and validation function.
    """
    # Setup warmup_epochs with fallback if None
    warmup_epochs = config.train.warmup_epochs if config.train.warmup_epochs is not None else 15
    
    # Setup optimizers
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    optimizer_ta = optim.AdamW(
        filter(lambda p: p.requires_grad, model_ta.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    # Setup schedulers
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.train.num_epochs,
        eta_min=1e-6
    )
    lr_scheduler_ta = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ta,
        T_max=config.train.num_epochs,
        eta_min=1e-6
    )
    
    # Initialize metrics
    train_dice = DiceMetric(include_background=True, reduction="mean")
    max_dice = -float('inf')
    best_epoch = 0
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        
        # Set models to training mode
        model.train()
        model_teacher.train()
        model_ta.train()
        
        train_metrics = {'dice': 0, 'sup_loss': 0, 'unsup_loss': 0}
        num_train = 0

        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        num_batches = len(train_loader['u_loader'])
        
        train_loop = tqdm(source_dataset, total=num_batches, 
                         desc=f'Epoch {epoch} Training', leave=False)
        train_dice.reset()

        for idx, (batch_l, batch_u) in enumerate(train_loop):
            i_iter = epoch * num_batches + idx
            
            # Get batch data
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()
            img_u = batch_u['img_w'].cuda().float()
            
            if epoch < warmup_epochs:
                # Warmup phase - supervised training only
                output = model(img_l)
                sup_loss = criterion[0](output, label_l)
                
                optimizer.zero_grad()
                sup_loss.backward()
                optimizer.step()
                
                # Forward pass for teacher models (no gradients)
                model_teacher.train()
                _ = model_teacher(img_l)
                model_ta.train()
                _ = model_ta(img_l)
                
                unsup_loss = torch.tensor(0.0).cuda()
                
            else:
                if epoch == warmup_epochs:
                    # Initialize teacher and TA models with student weights
                    with torch.no_grad():
                        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
                            t_params.data = s_params.data
                        for t_params, s_params in zip(model_ta.parameters(), model.parameters()):
                            t_params.data = s_params.data

                # Generate pseudo-labels using teacher model
                model_teacher.eval()
                with torch.no_grad():
                    pred_u_teacher = model_teacher(img_u)
                    pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
                    logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

                # Forward pass for all images
                num_labeled = len(img_l)
                image_all = torch.cat((img_l, img_u))
                
                # TA model forward pass
                pred_a_all = model_ta(image_all)
                pred_a_l, pred_a_u = pred_a_all[:num_labeled], pred_a_all[num_labeled:]

                # Teacher forward pass
                model_teacher.train()
                with torch.no_grad():
                    pred_all_teacher = model_teacher(image_all)
                    pred_u_teacher = pred_all_teacher[num_labeled:]

                # Calculate losses
                sup_loss = criterion[0](pred_a_l, label_l)
                
                # Unsupervised loss with confidence thresholding
                mask = logits_u_aug.ge(config.train.confidence_threshold).float()
                unsup_loss = (F.cross_entropy(pred_a_u, label_u_aug, reduction='none') * mask).mean()
                unsup_loss = unsup_loss * config.train.unsup_weight
                
                # Update TA model
                optimizer_ta.zero_grad()
                (sup_loss + unsup_loss).backward()
                optimizer_ta.step()

                # Update student model with EMA
                model.eval()
                with torch.no_grad():
                    ema_decay = min(1 - 1/(i_iter - len(train_loader['l_loader']) * warmup_epochs + 1), 0.999)
                    for t_params, s_params in zip(model.parameters(), model_ta.parameters()):
                        t_params.data = ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                model.train()

            # Update learning rates
            lr_scheduler.step()
            lr_scheduler_ta.step()
            
            # Update metrics
            batch_size = img_l.size(0)
            train_metrics['sup_loss'] = (train_metrics['sup_loss'] * num_train + sup_loss.item() * batch_size) / (num_train + batch_size)
            train_metrics['unsup_loss'] = (train_metrics['unsup_loss'] * num_train + unsup_loss.item() * batch_size) / (num_train + batch_size)
            num_train += batch_size

            # Update progress bar
            train_loop.set_postfix({
                'Sup Loss': f"{sup_loss.item():.4f}",
                'Unsup Loss': f"{unsup_loss.item():.4f}",
                'LR': f"{lr_scheduler.get_last_lr()[0]:.6f}"
            })

            # Log to file
            file_log.write(
                f"Epoch {epoch}, Iter {idx}, "
                f"Sup Loss: {sup_loss.item():.4f}, "
                f"Unsup Loss: {unsup_loss.item():.4f}\n"
            )
            file_log.flush()

        # Validation phase
        if epoch < warmup_epochs:
            metrics = validate_model(model, val_loader, criterion)
        else:
            metrics = validate_model(model_teacher, val_loader, criterion)
        
        # Update best model
        if metrics['dice'] > max_dice:
            max_dice = metrics['dice']
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch}! '
                      f'Dice: {metrics["dice"]:.4f}')
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        
        # Print epoch results
        epoch_time = time.time() - start
        log_message = (
            f"Epoch {epoch} completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s\n"
            f"Sup Loss: {train_metrics['sup_loss']:.4f}, "
            f"Unsup Loss: {train_metrics['unsup_loss']:.4f}\n"
            f"Validation Dice: {metrics['dice']:.4f}"
        )
        print(log_message)
        file_log.write(log_message + '\n')
        file_log.write('='*80 + '\n')
        file_log.flush()

        if config.debug:
            break
    
    print(f'Training completed. Best epoch: {best_epoch}')
    return model_teacher

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
    
    Args:
        config: Test configuration
        model: Model to test
        model_dir: Path to saved model weights
        test_loader: Test data loader
        criterion: Loss function(s)
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
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
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
    
    folds_to_train = [2,3,4,5]
    
    for fold in folds_to_train:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Update paths for each fold
        # exp_dir = '{}/{}_{}/fold{}'.format(config.data.save_folder, args.exp, config['data']['supervised_ratio'], fold)
        exp_dir = '{}/{}/fold{}'.format(config.data.save_folder, args.exp, fold)
        
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = '{}/best.pth'.format(exp_dir)
        test_results_dir = '{}/test_results.txt'.format(exp_dir)

        # Store yml file for each fold
        if config.debug == False:
            yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
            
        file_log = open('{}/log.txt'.format(exp_dir), 'w')
        
        # Train the model for this fold
        main(config)
        
        # Close the log file
        file_log.close()
        
        # Clear GPU memory between folds
        torch.cuda.empty_cache()
