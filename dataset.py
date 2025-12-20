
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # Import OpenCV for the border_mode constant

# (Phase A.3) Standard ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class GajarDataset(Dataset):
    """
    Custom PyTorch Dataset for the Gajar grades.
    Reads images and applies albumentations transforms.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Find all images and assign labels
        self.class_to_idx = {"Grade1": 0, "Grade2": 1, "Grade3": 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        for class_name, label_idx in self.class_to_idx.items():
            search_path = os.path.join(data_dir, class_name, "*.*")
            paths = glob.glob(search_path)
            self.image_paths.extend(paths)
            self.labels.extend([label_idx] * len(paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self)) # simple skip

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def create_sampler(dataset):
    """
    (Phase 0) Create a WeightedRandomSampler to handle class imbalance.
    """
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def get_transforms(config, is_train=True):
    """
    (Phase A) Create the augmentation pipeline.
    *** CORRECTED: Replaced GridMask with CoarseDropout ***
    """
    input_size = config['input_size']
    
    if is_train:
        # Full training pipeline from Phase A
        transform = A.Compose([
            
            # --- (Phase A.1) Manual Geometric & Color Transforms ---
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=10, 
                p=0.5, 
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.ColorJitter(
                brightness=0.08, 
                contrast=0.08, 
                saturation=0.08, 
                p=0.3
            ),

            # Aspect-ratio preserving resize + reflection pad
            A.LongestMaxSize(max_size=input_size, p=1.0),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=cv2.BORDER_REFLECT_101, # Reflection Padding
                p=1.0
            ),
            
            # (Phase A.2) External information dropping
            # *** THIS IS THE FIX: Replaced GridMask with CoarseDropout ***
            A.CoarseDropout(
                max_holes=8,          # Number of holes
                max_height=32,        # Size of holes (small)
                max_width=32,         # Size of holes (small)
                min_holes=4,
                fill_value=0,         # Fill with black
                p=config['gridmask_p'] # Use the same probability
            ),
            # --- End of Fix ---
            
            # (Phase A.3) Normalize
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(), # Convert numpy array to PyTorch tensor
        ])
    else:
        # Validation/Test pipeline (deterministic)
        transform = A.Compose([
            A.LongestMaxSize(max_size=input_size, p=1.0),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    return transform

def get_dataloaders(config):
    """
    Main function to create and return all three dataloaders
    """
    train_dir = os.path.join(config['data_dir'], 'train')
    valid_dir = os.path.join(config['data_dir'], 'valid')
    test_dir = os.path.join(config['data_dir'], 'test')

    # Get transforms
    train_transform = get_transforms(config, is_train=True)
    val_test_transform = get_transforms(config, is_train=False)

    # Create datasets
    train_dataset = GajarDataset(train_dir, transform=train_transform)
    val_dataset = GajarDataset(valid_dir, transform=val_test_transform)
    test_dataset = GajarDataset(test_dir, transform=val_test_transform)
    
    print(f"Dataset created. Train: {len(train_dataset)}, Valid: {len(val_dataset)}, Test: {len(test_dataset)}")

    # (Phase 0) Create the sampler for the training set
    train_sampler = create_sampler(train_dataset)
    
    print(f"WeightedRandomSampler created to handle imbalance.")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['physical_batch_size'],
        sampler=train_sampler, # Use the sampler
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True # Good for stable training
    )
    
    # Use a larger batch size for val/test since no backprop is needed
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['physical_batch_size'] * 2,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['physical_batch_size'] * 2,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
