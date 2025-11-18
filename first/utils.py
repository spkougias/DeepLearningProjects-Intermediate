import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar10_loaders(batch_size=64, augment=True, valid_size=0.1):
    """
    Creates 3 Data Loaders:
    1. Train: For updating weights (Standard Augmentation)
    2. Val:   For tuning/scheduler (No Augmentation) -> IMPROVES MODEL
    3. Test:  For final evaluation/plotting (No Augmentation) -> OBSERVATION ONLY
    """
    
    # 1. Define Transforms
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

    # Validation and Test should NOT have random augmentation
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # 2. Download Datasets
    # We load training data twice to apply different transforms to Train vs Val split
    train_set_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_set_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=eval_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=eval_transform
    )

    # 3. Create Split Indices
    num_train = len(train_set_full)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    # Shuffle to ensure random split
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # 4. Create Subsets
    train_data = Subset(train_set_full, train_idx)
    val_data = Subset(val_set_full, valid_idx)

    print(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_set)}")

    # 5. Create Loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
               
    return train_loader, val_loader, test_loader, classes