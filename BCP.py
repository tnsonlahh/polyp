import os
import time
import argparse
from datetime import datetime
from itertools import cycle
from skimage.measure import label
import logging

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.utils import DotDict, fix_all_seed
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        if input_tensor.ndim == 4 and input_tensor.shape[1] == self.n_classes:
            return input_tensor
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # Convert target to one-hot if it's not already
        if target.ndim == 3 or (target.ndim == 4 and target.shape[1] == 1):
            target = self._one_hot_encoder(target)
            
        assert inputs.size() == target.size(), f'predict & target shape do not match: inputs={inputs.size()}, target={target.size()}'
        
        if weight is None:
            weight = [1] * self.n_classes
            
        class_wise_dice = []
        loss = 0.0
        
        if mask is not None:
            mask = mask.repeat(1, self.n_classes, 1, 1) if mask.shape[1] != self.n_classes else mask
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
                
        return loss / self.n_classes
    
def save_model(net, path, metrics=None):
    """Save model state dict with optional metrics."""
    if metrics is None:
        torch.save(net.state_dict(), str(path))
    else:
        checkpoint = {
            'model': net.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, str(path))

def save_optimizer(optimizer, path):
    """Save only optimizer state dict."""
    torch.save(optimizer.state_dict(), str(path))

def load_model(net, path):
    """Load model state dict, handling both old and new checkpoint formats."""
    checkpoint = torch.load(str(path))
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # New format with metrics
        net.load_state_dict(checkpoint['model'])
    else:
        # Old format: just state dict
        net.load_state_dict(checkpoint)

def load_optimizer(optimizer, path):
    """Load only optimizer state dict."""
    optimizer.load_state_dict(torch.load(str(path)))

def create_model(ema=False):
    """
    Create a new model instance with EMA support.
    """
    model = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
    
def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, config):
    """Calculate the consistency weight for the current epoch."""
    warmup_epoch = config.train.warmup_epoch if getattr(config.train, 'warmup_epoch', None) is not None else 15
    return config.semi.conf_thresh * sigmoid_rampup(epoch, warmup_epoch)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    # Tạo mask với batch_size
    mask = torch.ones(batch_size, channel, img_x, img_y).cuda()  # Thêm batch_size dimension
    loss_mask = torch.ones(batch_size, channel, img_x, img_y).cuda()
    
    patch_x, patch_y = int(img_x*1/4), int(img_y*1/4)  
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    
    # Áp dụng mask cho mỗi sample trong batch
    mask[:, :, w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, :, w:w+patch_x, h:h+patch_y] = 0
    
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    dice_loss = DiceLoss(n_classes=3)
    output_soft = F.softmax(output, dim=1)  # [B, 3, H, W]

    # Đảm bảo img_l và patch_l có đúng kích thước
    if img_l.ndim == 3:  # Nếu là [B, H, W]
        img_l = F.one_hot(img_l.long(), num_classes=3).permute(0, 3, 1, 2).float()  # Chuyển về [B, 3, H, W]
    if patch_l.ndim == 3:
        patch_l = F.one_hot(patch_l.long(), num_classes=3).permute(0, 3, 1, 2).float()

    # Đảm bảo mask có đúng kích thước
    if mask.ndim == 3:  # [B, H, W]
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
    mask_dice = mask.repeat(1, 3, 1, 1) if mask.shape[1] == 1 else mask

    # Set weights
    image_weight, patch_weight = (u_weight, l_weight) if unlab else (l_weight, u_weight)

    # Calculate losses
    loss_dice = dice_loss(output_soft, img_l, mask_dice) * image_weight
    loss_ce = (CE(output, img_l.argmax(dim=1)) * mask[:, 0]).sum() / (mask[:, 0].sum() + 1e-16) * image_weight

    return loss_dice, loss_ce

def validate_model(model, val_loader):
    """
    Validate model using MONAI metrics.
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize MONAI metrics
    num_classes = 3
    if hasattr(config, 'model') and getattr(config.model, 'num_classes', None) is not None:
        num_classes = config.model.num_classes
    dice_metric = DiceMetric(include_background=True, num_classes=num_classes, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]
        
        with torch.no_grad():
            output = model(img)

            # Compute validation CE loss against class indices.
            if label.ndim == 4 and label.shape[1] > 1:
                label_idx = label.argmax(dim=1).long()
            elif label.ndim == 4 and label.shape[1] == 1:
                label_idx = label.squeeze(1).long()
            else:
                label_idx = label.long()
            batch_loss = F.cross_entropy(output, label_idx)
            metrics['loss'] = (metrics['loss'] * num_val + batch_loss.item() * batch_len) / (num_val + batch_len)
         
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
            
            # Update metrics
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)
            

            num_val += batch_len
            
            val_loop.set_postfix({
                'Dice': f"{dice_metric.aggregate().item():.4f}"
            })
    
    # Aggregate metrics
    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()
    
    # Reset metrics
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
    return metrics

def pre_train(config, snapshot_path, file_log):
    """
    Pre-training phase using config parameters
    """
    base_lr = config.train.optimizer.adamw.lr
    max_epoch = config.train.warmup_epoch if getattr(config.train, 'warmup_epoch', None) is not None else 15
    
    # Fix GPU setting
    gpu = getattr(config, 'gpu', '0')
    if gpu is None:
        gpu = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    # Setup model paths
    best_model_path = os.path.join(snapshot_path, 'pre_train_best.pth')
    best_optim_path = os.path.join(snapshot_path, 'pre_train_optim.pth')
    
    # Setup logging
    logging.basicConfig(filename=os.path.join(snapshot_path, 'training.log'),
                       level=logging.INFO,
                       format='%(asctime)s - %(message)s')
    
    # Create model and move to GPU
    model = create_model()
    model = model.cuda()

    def worker_init_fn(worker_id):
        random.seed(config.seed + worker_id)

    # Setup datasets and dataloaders (use supervised_ratio for labeled split)
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment4
    )
    
    trainloader = {
        'l_loader': DataLoader(
            dataset['lb_dataset'],
            batch_size=config.train.l_batchsize,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True
        ),
        'u_loader': DataLoader(
            dataset['ulb_dataset'],
            batch_size=config.train.u_batchsize,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True
        )
    }

    valloader = DataLoader(
        dataset['val_dataset'],
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    print(f"Unlabeled batches: {len(trainloader['u_loader'])}, Labeled batches: {len(trainloader['l_loader'])}")

    # Setup optimizer and metrics
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    train_dice = DiceMetric(include_background=True, reduction="mean")

    logging.info("Starting pre-training")
    
    model.train()
    iter_num = 0
    best_performance = 0.0
    early_stop_patience = config.train.early_stop_patience if getattr(config.train, 'early_stop_patience', None) is not None else 20
    early_stop_min_epochs = config.train.early_stop_min_epochs if getattr(config.train, 'early_stop_min_epochs', None) is not None else 30
    no_improve_epochs = 0
    
    # Sử dụng số epoch từ config (fallback nếu warmup_epoch chưa được set)
    max_epoch = config.train.warmup_epoch if getattr(config.train, 'warmup_epoch', None) is not None else 15
    
    for epoch in range(max_epoch):
        start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {
            'dice': 0,
            'loss': 0
        }
        num_train = 0
        
        source_dataset = zip(cycle(trainloader['l_loader']), trainloader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        train_dice.reset()
        
        for idx, (batch_l, batch_u) in enumerate(train_loop):
            # Get labeled data
            volume_batch, label_batch = batch_l['image'].cuda(), batch_l['label'].cuda()
            img_a, img_b = volume_batch[:config.train.l_batchsize//2], volume_batch[config.train.l_batchsize//2:]
            lab_a, lab_b = label_batch[:config.train.l_batchsize//2], label_batch[config.train.l_batchsize//2:]
            
            # Generate masks and mixed inputs
            img_mask, loss_mask = generate_mask(img_a)
            net_input = img_a * img_mask + img_b * (1 - img_mask)

            # Forward pass and loss calculation
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                output_soft = F.softmax(out_mixl, dim=1)
                output_onehot = torch.zeros_like(output_soft)
                output_onehot.scatter_(1, output_soft.argmax(dim=1, keepdim=True), 1)
                
                # Update MONAI metrics
                train_dice(y_pred=output_onehot, y=lab_a)
                
                # Update loss metric
                batch_len = img_a.shape[0]
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * batch_len) / (num_train + batch_len)
                num_train += batch_len

            # Update progress bar
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}"
            })

            iter_num += 1
            
            if config.debug:
                break

        # Get final training metrics for the epoch
        train_metrics['dice'] = train_dice.aggregate().item()
        train_dice.reset()

        # Log training metrics
        log_message = (f'Epoch {epoch}, Total train steps {idx} || '
                      f'Loss: {train_metrics["loss"]:.5f}, '
                      f'Dice: {train_metrics["dice"]:.4f}')
        print(log_message)
        file_log.write(log_message + '\n')
        file_log.flush()

        # Validation phase
        model.eval()
        metrics = validate_model(model, valloader)
        performance = metrics['dice']

        if performance > best_performance:
            best_performance = performance
            save_model(model, best_model_path, metrics)
            save_optimizer(optimizer, best_optim_path)
            message = f'Saved new best model | Dice: {metrics["dice"]:.4f}, IoU: {metrics["iou"]:.4f}, HD: {metrics["hd"]:.4f}, Loss: {metrics["loss"]:.5f}'
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch >= early_stop_min_epochs and no_improve_epochs > early_stop_patience:
            early_message = (f'Early stopping pre-train at epoch {epoch} after '
                             f'{no_improve_epochs} epochs without improvement.')
            print(early_message)
            file_log.write(early_message + '\n')
            file_log.flush()
            break

        model.train()

        # Log epoch time
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s\n')
        print('='*80)
        
        if config.debug:
            break

def self_train(config, pre_snapshot_path, snapshot_path, file_log):
    """
    Self-training phase using config parameters
    """
    base_lr = config.train.optimizer.adamw.lr
    warmup_epoch = config.train.warmup_epoch if getattr(config.train, 'warmup_epoch', None) is not None else 15
    max_epoch = max(1, config.train.num_epochs - warmup_epoch)
    
    # Fix GPU setting
    gpu = getattr(config, 'gpu', '0')
    if gpu is None:
        gpu = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    # Setup model paths
    pre_trained_model_path = os.path.join(pre_snapshot_path, 'pre_train_best.pth')
    pre_trained_optim_path = os.path.join(pre_snapshot_path, 'pre_train_optim.pth')
    best_model_path = os.path.join(snapshot_path, 'self_train_best.pth')
    best_optim_path = os.path.join(snapshot_path, 'self_train_optim.pth')

    # Create models and move to GPU
    model = create_model()
    model = model.cuda()
    ema_model = create_model(ema=True)
    ema_model = ema_model.cuda()

    def worker_init_fn(worker_id):
        random.seed(config.seed + worker_id)

    # Setup datasets (use supervised_ratio for labeled split)
    dataset = get_dataset(
        config,
        img_size=config.data.img_size,
        supervised_ratio=config.data.get('supervised_ratio', 0.2),
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment4
    )
    
    trainloader = {
        'l_loader': DataLoader(
            dataset['lb_dataset'],
            batch_size=config.train.l_batchsize,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True
        ),
        'u_loader': DataLoader(
            dataset['ulb_dataset'],
            batch_size=config.train.u_batchsize,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True
        )
    }

    valloader = DataLoader(
        dataset['val_dataset'],
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    print(f"Unlabeled batches: {len(trainloader['u_loader'])}, Labeled batches: {len(trainloader['l_loader'])}")

    # Setup optimizer first so we can load its state
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )

    # Load pre-trained weights and optimizer state
    load_model(ema_model, pre_trained_model_path)
    load_model(model, pre_trained_model_path)
    load_optimizer(optimizer, pre_trained_optim_path)
    
    logging.info("Loaded pre-trained model from {}".format(pre_trained_model_path))
    logging.info("Loaded pre-trained optimizer from {}".format(pre_trained_optim_path))

    logging.info("Start self-training")
    logging.info("{} iterations per epoch".format(len(trainloader['l_loader'])))

    model.train()
    ema_model.train()
    iter_num = 0
    best_performance = 0.0
    early_stop_patience = config.train.early_stop_patience if getattr(config.train, 'early_stop_patience', None) is not None else 10
    early_stop_min_epochs = config.train.early_stop_min_epochs if getattr(config.train, 'early_stop_min_epochs', None) is not None else 30
    no_improve_epochs = 0
    
    # Tính số epoch cho self-training
    max_epoch = max(1, config.train.num_epochs - warmup_epoch)

    for epoch in range(max_epoch):
        source_dataset = zip(cycle(trainloader['l_loader']), trainloader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        
        for idx, (batch_l, batch_u) in enumerate(train_loop):
            volume_batch, label_batch = batch_l['image'].cuda(), batch_l['label'].cuda()
            unlabeled_batch = batch_u['img_w'].cuda()

            # Split labeled data
            img_a, img_b = volume_batch[:config.train.l_batchsize//2], volume_batch[config.train.l_batchsize//2:config.train.l_batchsize]
            uimg_a, uimg_b = unlabeled_batch[:config.train.u_batchsize//2], unlabeled_batch[config.train.u_batchsize//2:]
            lab_a, lab_b = label_batch[:config.train.l_batchsize//2], label_batch[config.train.l_batchsize//2:config.train.l_batchsize]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = torch.softmax(pre_a, dim=1) 
                plab_b = torch.softmax(pre_b, dim=1)


                img_mask, loss_mask = generate_mask(img_a)
                unl_label = plab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + plab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(iter_num//150, config)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l = model(net_input_l)
            
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=consistency_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=consistency_weight)

            loss_ce = unl_ce + l_ce
            loss_dice = unl_dice + l_dice
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, config.train.ema_decay)

            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{loss_dice.item():.4f}"
            })

            # Validation mỗi epoch

        model.eval()
        metrics = validate_model(model, valloader)
        performance = metrics['dice']

        if performance > best_performance:
            best_performance = performance
            save_model(model, best_model_path, metrics)
            save_optimizer(optimizer, best_optim_path)
            message = f'Saved new best model | Dice: {metrics["dice"]:.4f}, IoU: {metrics["iou"]:.4f}, HD: {metrics["hd"]:.4f}, Loss: {metrics["loss"]:.5f}'
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch >= early_stop_min_epochs and no_improve_epochs > early_stop_patience:
            early_message = (f'Early stopping self-train at epoch {epoch} after '
                             f'{no_improve_epochs} epochs without improvement.')
            print(early_message)
            file_log.write(early_message + '\n')
            file_log.flush()
            break

        model.train()


    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with FixMatch')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--consistency', type=float, default=0.5)
    parser.add_argument('--consistency_rampup', type=float, default=75.0)
    
    args = parser.parse_args()
    
    # Load and update config
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
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
        pre_snapshot_path = exp_dir  # Path to pre-trained model directory
        snapshot_path = exp_dir      # Path to save self-training models
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save config
        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
        
        # Train fold
        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            pre_train(config, exp_dir, file_log)
            self_train(config, exp_dir, exp_dir, file_log)
            
        torch.cuda.empty_cache()
