import torch
import torch.nn as nn

class AE_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoding layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ) # N,16,28,28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        ) # N,8,14,14

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 7 * 7, 2),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2, 8 * 7 * 7),
            nn.BatchNorm1d(8 * 7 * 7),
            nn.ReLU(),
        )

        # Decoding layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.conv2(x2)
        x4 = self.pool(x3)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.fc1(x4)

        # Decoder
        x6 = self.fc2(x5)
        x7 = x6.view(x6.size(0), 8, 7, 7)
        x8 = self.conv3(x7)
        x9 = self.up_sample(x8)
        out = self.conv4(x9)

        return out
    
    def get_latent_code(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.conv2(x2)
        x4 = self.pool(x3)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.fc1(x4)
        return x5