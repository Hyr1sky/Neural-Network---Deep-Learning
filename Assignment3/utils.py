import numpy as np
import torch

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