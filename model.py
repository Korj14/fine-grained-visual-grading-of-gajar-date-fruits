
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from dropblock import DropBlock2D

class FGVCModel(nn.Module):
    """
    This class wraps the timm EfficientNet-B3 model and integrates:
    1. (Phase B) DropBlock in the classifier head.
    2. (Phase C) Manifold MixUp logic in the forward pass.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 1. Load Backbone ---
        self.backbone = timm.create_model(
            config['model_name'], 
            pretrained=True,
            num_classes=0,       # Remove classifier
            global_pool=''     # Remove global pooling
        )
        
        backbone_out_features = self.backbone.num_features
        
        # --- 2. Create Head (with DropBlock) ---
        # (Phase B) DropBlock MUST come before pooling and flattening
        self.head = nn.Sequential(
            DropBlock2D(
                block_size=config['dropblock_block_size'],
                drop_prob=0.0 # Start at 0, will be scheduled by the trainer
            ),
            nn.AdaptiveAvgPool2d(1), # This is the Global Pooling
            nn.Flatten(),
            nn.Linear(backbone_out_features, config['num_classes'])
        )
        
        self.mixing_alpha = config['manifold_mixup_alpha']
        self.mixing_rate = config['manifold_mixup_batch_rate']

    def get_backbone(self):
        """Helper for the optimizer to set discriminative LRs"""
        return self.backbone
    
    def get_classifier(self):
        """Helper for the optimizer to set discriminative LRs"""
        return self.head

    def forward(self, x, targets=None):
        """
        Main forward pass.
        Handles Manifold MixUp logic.
        """
        
        # --- 1. Backbone Pass ---
        # Get the feature maps from the backbone
        x = self.backbone(x) 

        # --- 2. Manifold MixUp (Phase C) ---
        # *** THIS IS THE FIX ***
        # Check if we should apply MixUp
        if self.training and self.config['use_manifold_mixup'] and targets is not None and torch.rand(1).item() <= self.mixing_rate:
            
            # Get mixing lambda from Beta distribution
            lam = np.random.beta(self.mixing_alpha, self.mixing_alpha)
            
            # Shuffle the batch to get mixing pairs
            batch_size = x.size(0)
            indices = torch.randperm(batch_size, device=x.device)
            
            # Mix the feature maps (x)
            x = lam * x + (1 - lam) * x[indices, :]
            
            # Mix the targets (labels)
            targets_onehot = F.one_hot(targets, num_classes=self.config['num_classes']).float()
            mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[indices, :]
            
            # Pass mixed features to the head
            logits = self.head(x)
            
            # Return logits and *mixed* targets for the loss function
            return logits, mixed_targets
        
        # --- 3. Standard Head Pass (No MixUp) ---
        # If not training, or not applying mixup this batch
        logits = self.head(x)
        
        if targets is not None:
            # For standard training/validation loss
            return logits, targets
        
        # For inference
        return logits
