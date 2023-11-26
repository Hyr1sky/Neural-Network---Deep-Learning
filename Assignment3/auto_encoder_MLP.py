import torch
import torch.nn as nn
from add_noise import add_noise

class AE_MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # [b, 784] => [b, 2]
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        # [b, 2] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size, 784)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batch_size, 1, 28, 28)

        return x
    
    def get_latent_code(self, x):
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size, 784)
        # encoder
        x = self.encoder(x)
        return x
    
    def generate_img(self, x):
        batch_size = x.size(0)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batch_size, 1, 28, 28)
        return x
    