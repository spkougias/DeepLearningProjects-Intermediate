import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# activation
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"404 Activation not found")


# model init
def initialize_model(model_name, dropout_rate, device, activation='relu'):

    if model_name == 'simple_mlp':
        model = SimpleMLP(dropout_rate=dropout_rate, activation=activation)
    elif model_name == 'complex_mlp':
        model = ComplexMLP(dropout_rate=dropout_rate, activation=activation)
    elif model_name == 'cnn':
        model = CNN(dropout_rate=dropout_rate, activation=activation)
    elif model_name == 'transfer':
        model = TransferLearningModel(dropout_rate=dropout_rate, activation=activation)
    else:
        raise ValueError("404 Model Not Found")
    
    return model.to(device)


# simple_mlp
class SimpleMLP(nn.Module):
    def __init__(self, dropout_rate=0.0, activation='relu'): 
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 10)
        self.act = get_activation(activation)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# complex_mlp
class ComplexMLP(nn.Module):
    def __init__(self, dropout_rate=0.5, activation='relu'):
        super(ComplexMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.activation = get_activation(activation)

        self.fc1 = nn.Linear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x



# cnn
class CNN(nn.Module):
    def __init__(self, dropout_rate=0.5, activation='relu'):
        super(CNN, self).__init__()
        self.act = get_activation(activation)
        
        conv_dropout = dropout_rate * 0.5

        # block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(conv_dropout)

        # block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(conv_dropout)

        # fully connected
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout2(x)

        x = x.view(-1, 128 * 8 * 8)

        x = self.act(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# transfer learning (ResNet18)
class TransferLearningModel(nn.Module):
    def __init__(self, dropout_rate=0.0, activation='relu'): 
        super(TransferLearningModel, self).__init__()
        #load weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #modify for 32x32 images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove the first maxpool entirely
        
        # unfreeze
        for param in self.resnet.parameters():
            param.requires_grad = True 
            
        # replace classifier head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 10)
        )
    def forward(self, x):
        return self.resnet(x)


