import torch
import torch.nn as nn
import torch.utils.data as data
import time
import os
import numpy as np
from utils import ShowImg
from utils import Progress_LatendCodes
from auto_encoder_MLP import AE_MLP
from auto_encoder_CNN import AE_CNN
from add_noise import add_noise
from torchvision import transforms
from auto_encoder import AutoEncoderModel
from torchvision.utils import save_image
from torchvision.datasets import MNIST

# Constant
EPOCH = 30
BATCH_SIZE = 100
LR = 0.001
WEIGHT_DECAY = 0


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.add_noise = add_noise(std=1)
        # self.net = AutoEncoderModel().to(self.device)
        # self.net = AE_CNN().to(self.device)
        self.net = AE_MLP().to(self.device)
        self.loss_fn = nn.MSELoss() # BCE requires the input to be in the range of [0,1]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.trans = transforms.Compose([transforms.ToTensor(), ])

    def train(self):
        # Set up the directory
        if not os.path.exists("./Assignment3/params"):
            os.mkdir("./Assignment3/params")
        if not os.path.exists("./Assignment3/img"):
            os.mkdir("./Assignment3/img")
            
        epoch = EPOCH
        batch_size = BATCH_SIZE

        Train_Data = MNIST(root="./Assignment3_dataset/MNIST", train=True, download=False, transform=self.trans)
        Train_DataLoader = data.DataLoader(dataset=Train_Data, shuffle=True, batch_size=batch_size)

        Latend_Codes = []
        Labels = []

        # print the structure of the network
        print(self.net)

        for epochs in range(epoch):
            print("-------------No.{} Round Training--------------".format(epochs + 1))
            # check time
            start_time = time.time()

            for i, (x, y) in enumerate(Train_DataLoader):

                # add noise
                # x = self.add_noise(x)
                noise_factor = 0.4
                x = x + torch.normal(mean=0.0, std=1, size=x.size()) * noise_factor

                # process
                img = x.to(self.device)
                label = y.to(self.device)
                L2_Reg = 0.0
                for param in self.net.parameters():
                    L2_Reg += torch.norm(param, 2)
                out_img = self.net(img)
                loss = self.loss_fn(out_img, img) + WEIGHT_DECAY * L2_Reg

                # backprop
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

            # Save the reconstructed images
            generative_images = out_img.cpu().data
            real_images = img.cpu().data
            five_generative_images = generative_images[0:5]
            five_real_images = real_images[0:5]
            all_images = torch.cat((five_real_images, five_generative_images), 0)
            save_image(all_images, "./Assignment3/img/SingleGen/contract_img_{}.png".format(epochs + 1), nrow=5)
            # ShowImg(all_images, epochs + 1, "contract_img_", "Assignment3/img/SingleGen/", False)
            save_image(generative_images, "./Assignment3/img/generative_images_{}.png".format(epochs + 1), nrow=10)
            save_image(real_images, "./Assignment3/img/real_images_{}.png".format(epochs + 1), nrow=10)

        # Save the model
        torch.save(self.net.state_dict(), "./Assignment3/params/net.pth")
        Progress_LatendCodes(Latend_Codes, Labels, False)

    def generate(self):
        """
        Select an area of the latent space
        Generate images from the selected area
        """
        
        if not os.path.exists("./Assignment3/img/Generate"):
            os.mkdir("./Assignment3/img/Generate")

        # Load the model
        self.net.load_state_dict(torch.load("./Assignment3/params/net.pth"))

        # Generate images
        # Set a range of latent codes
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)

        x, y = np.meshgrid(x, y)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = np.concatenate((x, y), axis=1)
        z = torch.from_numpy(z).float().to(self.device)
        generated_images = self.net.generate_img(z)
        save_image(generated_images, "./Assignment3/img/Generate/generated_images.png", nrow=20)
        # ShowImg(generated_images, 0, "generated_images", "Assignment3/img/Generate/", False)

            
if __name__ == '__main__':
    t = Trainer()
    t.train()
    t.generate()