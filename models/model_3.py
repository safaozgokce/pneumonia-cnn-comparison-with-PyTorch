# model_3 - https://www.kaggle.com/code/rajasreerajamohanan/pneumonia-detection


import torch
from torch import nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self, img_width=224, img_height=224, num_classes=2):
        super().__init__()
        
        # Block-1 (Giriş: 1 kanal - grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block-2
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block-3
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.pool3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Boyut hesaplama
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, img_height, img_width)  # 1 kanal
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2_1(F.relu(self.conv2_1(x)))
            x = self.pool2_2(F.relu(self.conv2_2(x)))
            x = self.pool3_1(F.relu(self.conv3_1(x)))
            x = self.pool3_2(F.relu(self.conv3_2(x)))
            flattened_size = x.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Block-1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Block-2
        x = F.relu(self.conv2_1(x))
        x = self.pool2_1(x)
        x = F.relu(self.conv2_2(x))
        x = self.pool2_2(x)
        
        # Block-3
        x = F.relu(self.conv3_1(x))
        x = self.pool3_1(x)
        x = F.relu(self.conv3_2(x))
        x = self.pool3_2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        return logits