import torch
from torchvision import datasets, transforms
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

mean = 0.
std = 0.

for images, _ in loader:
    # Shape of images: [Batch_Size, Channels, Height, Width]
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
mean /= 50000
std /= 50000

print(f"Mean: {mean}")
print(f"Std: {std}")   