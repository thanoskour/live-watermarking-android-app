from skimage.metrics import structural_similarity as ssim
from math import sqrt
import math
from PIL import Image
import itertools as it
import numpy as np
from math import log10, sqrt

def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0):  # MSE is zero means no noise is present in the signal .
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def ssim_metric(img1,img2):
	value = ssim(img1, img2,data_range = img2.max() - img2.min(), channel_axis = -1)
	return value

import sys


	