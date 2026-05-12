# model_0 (kendi modelim)

import torch
from torch import nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Birinci evrişimli blok
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # İkinci evrişimli blok
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Üçüncü evrişimli blok
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Flatten layer ( çok boyutlu veri -> tek boyutlu bir vektör )
        self.flatten = nn.Flatten()
        
        # Sınıflandırma için tam bağlantılı (fully connected) katmanlar
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        # Girdiyi evrişimli bloklardan geçirme
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Öznitelikleri (features) düzleştirme
        x = self.flatten(x)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)
        
        # Sonuç skorlarını (logits) alıyoruz
        return logits