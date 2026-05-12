# model_2 - https://www.kaggle.com/code/akbarprdna/pneumonia-detection-from-chest-x-ray-using-cnn

import torch
from torch import nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.1)  # Conv'a dropout ekledik
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.15)
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout2d(0.2)
        
        # Global Average Pooling (KEY IMPROVEMENT)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # FC Layers with L2 regularization (PyTorch'ta weight_decay ile)
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv Blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 256, 1, 1) -> (batch, 256)
        
        # FC Layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        
        logits = self.fc3(x)
        
        return logits