from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(image1,image2):
	value = ssim(image1, image2, data_range = image2.max() - image2.min(), channel_axis = -1,multichannel=True)
	return value