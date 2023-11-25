import torch
import torch.nn as nn
import torch.utils.data as data
import time
import os
import numpy as np
from utils import ShowImg
from utils import Progress_LatendCodes
from add_noise import add_noise
from torchvision import transforms
from auto_encoder import AutoEncoderModel
from torchvision.utils import save_image
from torchvision.datasets import MNIST

# Constant
EPOCH = 10
BATCH_SIZE = 100
LR = 0.001
STD = 0.1

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AutoEncoderModel().to(self.device)
        self.loss_fn = nn.BCELoss() # BCE requires the input to be in the range of [0,1]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.trans = transforms.Compose([transforms.ToTensor(), ])

    def train(self):
        # Set up the directory
        if not os.path.exists("./Assignment3/params"):
            os.mkdir("./Assignment3/params")
        if not os.path.exists("./Assignment3/img"):
            os.mkdir("./Assignment3/img")
            
        epoch = EPOCH
        batch_size = BATCH_SIZE

        Train_Data = MNIST(root="../Assignment3_dataset", train=True, download=True, transform=self.trans)
        Train_DataLoader = data.DataLoader(dataset=Train_Data, shuffle=True, batch_size=batch_size)

        Latend_Codes = []
        Labels = []

        for epochs in range(epoch):
            print("-------------No.{} Round Training--------------".format(epochs + 1))
            # check time
            start_time = time.time()

            for i, (x, y) in enumerate(Train_DataLoader):

                # add noise
                noise_factor = 0.4
                x = x + noise_factor * torch.randn(*x.shape)

                img = x.to(self.device)
                label = y.to(self.device)
                out_img = self.net(img)
                loss = self.loss_fn(out_img, img)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # get latend code
                latent_code = self.net.get_latent_code(img)
                Latend_Codes.append(latent_code.detach().cpu().numpy())
                Labels.append(label.detach().cpu().numpy())
                
                if i % 100 == 0:
                    print("epochs: [{}], iteration: [{}]/[{}], loss: {:.3f}".format(epochs + 1, i, len(Train_DataLoader), loss.float()))
            
            # check time
            end_time = time.time()
            print("time cost : {:.3f} s".format(end_time - start_time))

            # Save the model
            torch.save(self.net.state_dict(), "./Assignment3/params/net.pth")

            # Save the reconstructed images
            generative_images = out_img.cpu().data
            real_images = img.cpu().data
            ShowImg(generative_images[0], epochs + 1, "gen_img_", "Assignment3/img/SingleGen/", False)
            ShowImg(real_images[0], epochs + 1, "real_img_", "Assignment3/img/SingleGen/", False)
            save_image(generative_images, "./Assignment3/img/generative_images_{}.png".format(epochs + 1), nrow=10)
            save_image(real_images, "./Assignment3/img/real_images_{}.png".format(epochs + 1), nrow=10)

        Progress_LatendCodes(Latend_Codes, Labels)

            
if __name__ == '__main__':
    t = Trainer()
    t.train()