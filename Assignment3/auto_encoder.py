import torch
import torch.nn as nn
from add_noise import add_noise

class AutoEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # N,3,14,14
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 2, 1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # N,6,7,7
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 12, 3, 2, 1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # N,12,4,4
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 24, 3, 2, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # N,24,2,2

        self.fc1 = nn.Sequential(
            # bottleneck
            nn.Linear(24 * 2 * 2, 2),
        )  # Origin N,128

        self.fc2 = nn.Sequential(
            # bottleneck
            nn.Linear(2, 24 * 2 * 2),
            nn.BatchNorm1d(24 * 2 * 2),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # Origin 7,7


        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(24, 12, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # nn.Sigmoid(),
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(12, 6, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # N,6,7,7
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(6, 3, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # nn.Sigmoid(),
        )  # N,3,14,14
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(3, 1, 3, 2, 1, output_padding=1),
            # nn.ReLU(),
            nn.Sigmoid(),
        )  # N,1,28,28

        self.resize = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)


    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x4.view(x4.size(0), -1)
        code = self.fc1(x4)

        # Decoder
        y1 = self.fc2(code)
        y1 = y1.view(y1.size(0), 24, 2, 2)
        y2 = self.conv5(y1)
        y3 = self.conv6(y2)
        y4 = self.conv7(y3)
        out = self.conv8(y4)
        out = self.resize(out)
        out = torch.clamp(out, 0, 1)

        return out
    

    def get_latent_code(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x4.view(x4.size(0), -1)
        code = self.fc1(x4)
        return code


    def generate_img(self, latent_code):
        with torch.no_grad():
            # Decoder
            y1 = self.fc2(latent_code)
            y1 = y1.view(y1.size(0), 24, 2, 2)
            y2 = self.conv5(y1)
            y3 = self.conv6(y2)
            y4 = self.conv7(y3)
            generated_image = self.conv8(y4)
        return generated_image

