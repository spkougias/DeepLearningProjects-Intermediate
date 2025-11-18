import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. Simple MLP (Multi-Layer Perceptron)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB).
        # Flattened input size = 32 * 32 * 3 = 3072
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 512) # Hidden Layer
        self.fc2 = nn.Linear(512, 10)   # Output Layer (10 classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x)) # Activation function
        x = self.fc2(x)
        return x

# ==========================================
# 2. Complex MLP
# ==========================================
class ComplexMLP(nn.Module):
    def __init__(self):
        super(ComplexMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Deeper network with Dropout to prevent overfitting
        self.fc1 = nn.Linear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024) # Batch Normalization helps training stability
        self.dropout1 = nn.Dropout(0.5) # Randomly turn off 50% of neurons
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 10) # Output

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# ==========================================
# 3. Simple CNN (Convolutional Neural Network)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolution 1: Input 3 channels -> Output 32 channels, kernel size 3x3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # Max Pool reduces size by half (32x32 -> 16x16)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolution 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Max Pool (16x16 -> 8x8)
        
        # Fully Connected layers
        # Input size calculation: 64 channels * 8 * 8 pixels
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 4. Complex CNN
# ==========================================
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Pool defined above is reused
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully Connected
        # After 2 pools, 32x32 becomes 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Dense
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# ==========================================
# 5. Transfer Learning (ResNet18)
# ==========================================
class TransferLearningModel(nn.Module):
    def __init__(self):
        super(TransferLearningModel, self).__init__()
        # Load a pre-trained ResNet18 model
        # 'pretrained=True' downloads weights trained on ImageNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze early layers so we don't ruin the pre-trained features
        # (Optional: for this assignment, fine-tuning everything often works better 
        # on small data like CIFAR, but strict transfer learning usually freezes layers)
        for param in self.resnet.parameters():
            param.requires_grad = False 
            
        # Replace the last layer (fc) to match CIFAR-10 classes (10 instead of 1000)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)