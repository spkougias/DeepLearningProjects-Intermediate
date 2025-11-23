import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10_loaders(batch_size=64, augment=True, val_size=0.1):
    
    # stats from get_stats script
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)
    
    # transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])

    # evaluation batch not transformed only normalised
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means,stds)
    ])

    train_set_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_set_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=eval_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=eval_transform
    )

    # split and shuffle the data
    len_train = len(train_set_full)
    indices = list(range(len_train))
    split = int(np.floor(val_size*len_train))
    
    np.random.shuffle(indices) # randomise training data
    train_idx, valid_idx = indices[split:], indices[:split]
    
    train_data = Subset(train_set_full, train_idx)
    val_data = Subset(val_set_full, valid_idx)

    print(f"Train= {len(train_data)}, Val= {len(val_data)}, Test= {len(test_set)}")

    # loaders
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
