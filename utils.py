
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os

class FocalLoss(nn.Module):
    """
    (Phase 0) Implements the Focal Loss.
    This replaces CrossEntropyLoss and the WeightedRandomSampler.
    It focuses training on "hard" examples (where pt is low) and
    down-weights the loss for "easy" examples.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Can be a tensor of weights, one for each class
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs.shape = (B, C), targets.shape = (B)
        
        # Calculate Cross-Entropy loss, but without the final reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities (pt) for the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate the Focal Loss
        # The main formula: (1 - pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Apply class-specific weighting
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_optimizer(model, config):
    """
    (Phase 0) Creates the AdamW optimizer with decoupled weight decay.
    """
    # (Phase D) Set up discriminative learning rates
    head_params = model.get_classifier().parameters()
    backbone_params = model.get_backbone().parameters()
    
    # --- THIS IS THE FIX: Cast all config values to float ---
    head_lr = float(config['head_lr'])
    backbone_lr = float(config['backbone_lr'])
    weight_decay = float(config['weight_decay'])
    
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]
    
    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            param_groups,
            lr=head_lr, # Default LR (will be set by group)
            weight_decay=weight_decay
        )
    else:
        # Fallback to SGD
        optimizer = optim.SGD(
            param_groups,
            lr=head_lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    return optimizer

def get_scheduler(optimizer, config, total_steps_per_epoch):
    """
    (Phase 0) Creates the Cosine Annealing scheduler with a linear warmup.
    """
    # --- THIS IS THE FIX: Cast config values to float/int ---
    total_epochs = int(config['total_epochs'])
    head_epochs = int(config['head_epochs'])
    warmup_epochs = int(config['warmup_epochs'])
    min_lr = float(config['min_lr'])
    
    # Total steps for the main cosine annealing part
    main_lr_steps = (total_epochs - head_epochs) * total_steps_per_epoch
    
    # Total steps for the warmup part (e.g., 3 epochs)
    warmup_steps = warmup_epochs * total_steps_per_epoch

    # 1. Warmup Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # 2. Main Cosine Scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=main_lr_steps - warmup_steps, # The steps *after* warmup
        eta_min=min_lr
    )
    
    # 3. Chain them together
    # Scheduler 1 (warmup) runs for 'warmup_steps'
    # Scheduler 2 (cosine) runs for the rest
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return lr_scheduler

def save_checkpoint(epoch, model, optimizer, scheduler, config, filename="checkpoint.pth.tar"):
    """
    (Phase F) Saves the complete training state.
    """
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }
    filepath = os.path.join(config['checkpoint_dir'], filename)
    torch.save(state, filepath)
    print(f"==> Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, scheduler, config, filename="checkpoint.pth.tar"):
    """
    (Phase F) Loads a training state to resume.
    """
    filepath = os.path.join(config['checkpoint_dir'], filename)
    start_epoch = 0
    if os.path.isfile(filepath):
        print(f"==> Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"==> Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
    else:
        print(f"==> No checkpoint found at '{filepath}'")
        
    return model, optimizer, scheduler, start_epoch
