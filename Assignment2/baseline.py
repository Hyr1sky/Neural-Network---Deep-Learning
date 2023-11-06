import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

train_transform = transforms.Compose([
    # 如果你不是选择的是PIL导入图片，还需要使用 transforms.ToPILImage()
    # 将图像固定为一个尺寸
    transforms.Resize((128, 128)),
    # 在此处添加五个及以上的变换

    # 最后一个transform必须是ToTensor.
    transforms.ToTensor(),
])

class CNN_baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )

    def forward(self, x):
        #
        out = self.CNN(x)

        return out