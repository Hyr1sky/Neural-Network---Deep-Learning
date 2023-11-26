import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def CalcuPSNR(img1, img2, max_val=255.):
    """
    Based on `tf.image.psnr`
    https://www.tensorflow.org/api_docs/python/tf/image/psnr
    """
    float_type = 'float64'
    # img1 = (torch.clamp(img1,-1,1).cpu().numpy() + 1) / 2 * 255
    # img2 = (torch.clamp(img2,-1,1).cpu().numpy() + 1) / 2 * 255
    img1 = torch.clamp(img1, 0, 1).cpu().numpy() * 255
    img2 = torch.clamp(img2, 0, 1).cpu().numpy() * 255
    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def MSE2PSNR(MSE):
    return 10 * math.log10(255 ** 2 / (MSE))

def ShowImg(img, idx, type, folder, show):
    img = img.cpu().data
    save_image(img, "./{}/{}{}.png".format(folder, type, idx))
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

def Progress_LatendCodes(Latend_Codes, Labels, show):
    Latend_Codes = np.concatenate(Latend_Codes, axis=0)
    Labels = np.concatenate(Labels, axis=0)
    np.save("./Assignment3/params/Latend_Codes.npy", Latend_Codes)
    np.save("./Assignment3/params/Labels.npy", Labels)

    if show:
        plt.figure(figsize=(10, 10))
        plt.scatter(Latend_Codes[:, 0], Latend_Codes[:, 1], c=Labels, cmap='gray')
        plt.colorbar()
        plt.savefig("./Assignment3/img/LatendCode/Latend_Codes.png")
        plt.show()