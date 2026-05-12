# model_4 - https://www.kaggle.com/code/abcdabcddsvgsgdsgf/nexpp

import torch
from torch import nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Conv Layer 1 (Giriş: 1 kanal - grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Boyut hesapla
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)  # 1 kanal
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            flattened_size = x.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits