#from .encodeinteger import encodeInteger
#from .decodesip import decodeSip
#from .embed_key import EmbedPermutation
#from .extract_sip import ExtractPermutation
import decodesip
import encodeinteger
import embed_key
import extract_sip
import recossip
#from .recossip import recsip
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import sqrt
import math
from PIL import Image
import itertools as it
import numpy as np
from math import log10, sqrt
from time import process_time
import cv2
import sys
import re

'''
KEY = int(sys.argv[1])
IMAGE = sys.argv[2]
IMAGE_NAME = re.split(r'\.(?!\d)', IMAGE)[0]
COMMAND = sys.argv[3]
COPT = float(sys.argv[4])
global SIP 
global SIZE
'''


def init():
	w = embed_key.EmbedPermutation()
	ex = extract_sip.ExtractPermutation()

	return w,ex

def mergeCellsToImage(cells,w,h,N,im_format,s):
	if(s == 3):
		display = np.empty(((w)*N, (h)*N , s), dtype = np.uint8)
		for i, j in it.product(range(N), range(N)):
			arr = np.array(cells[i*N+j])
			x,y = i*(w), j*(h)
			display[x : x + (w), y : y + (h)] = arr

	else:
		display = np.empty(((w)*N, (h)*N , 4), dtype = np.uint8)
		for i, j in it.product(range(N), range(N)):
			arr = np.array(cells[i*N+j])
			x,y = i*(w), j*(h)
			display[x : x + (w), y : y + (h)] = arr
			
	
	return display


def openImage(path):
		img = Image.open(path)
		return img


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def ssim_metric(img1,img2):
	image1 = Image.open(img1)
	image2 = Image.open(img2)
	image1 = np.array(image1)
	image2 = np.array(image2)
	value = ssim(image1, image2, data_range=image2.max() - image2.min(),multichannel=True)
	return value


def extract_recsip(im,keys,pr,pb,gs,k,im_format):
	em,ex = init()


	img = openImage(im)

	w,h = img.size

	M,N = img.size
	channel_array = np.array(img)


	grid_size_w = math.floor((N / gs))
	grid_size_h = math.floor((M / gs))

	cells = []
	sips = []
	extracted = []
	ex_sips = []

	i = 0
	c_index = 0
	for r in range(0,N - grid_size_w + 1, grid_size_w):
		for c in range(0,M - grid_size_h + 1, grid_size_h):

			grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]
			cells.append(grid_cell)
			g_cell = Image.fromarray(grid_cell)
			c_sip = encodeinteger.encodeInteger(int(keys[i]))
			size = len(c_sip)
			sip1,sip2,sip3 = ex.getSip(g_cell,size,k,pr,pb,im_format)
			c_index += 1
			sips.append((sip1,sip2,sip3))

	#im = np.empty((N,M,3),dtype = np.uint8)

	not_ex = 0
	code = []
	for i in range(len(sips)):
		s1 = sips[i][0]
		s2 = sips[i][1]
		s3 = sips[i][2]
		ex_sips.append(("block" + str(i + 1),s1,s2,s3))
		c_sip = encodeinteger.encodeInteger(int(keys[i]))
		if(s1 == c_sip): 
			k1 = decodesip.decodeSip(s1)
			extracted.append(("block" + str(i + 1),k1))
			code.append(k1)
			print("Extracted key: ",k1)
		elif(s2 == c_sip):
			k2 = decodesip.decodeSip(s2)
			extracted.append(("block" + str(i + 1),k2)) 
			print("Extracted key: ",k2)
			code.append(k2)
		elif(s3 == c_sip):
			k3 = decodesip.decodeSip(s3)
			extracted.append(("block" + str(i + 1),k3)) 
			print("Extracted key: ",k3)
			code.append(k3)
			
		else:
			extracted.append(("block" + str(i + 1),'X'))
			print("Extracted key: X")
			code.append('X')
			c_cell = cells[i] 
			c_cell[:,:,0] = 255
			not_ex += 1

	succ_rate = (1 - (not_ex / (gs**2))) * 100

	return extracted,ex_sips,succ_rate,code
			





def recursive_embed(im,keys,k,IMAGE_NAME,COPT,pr,pb,gs,im_format,root_path):


	em,ex = init()
	img = openImage(im)
	w,h = img.size
	M,N = img.size
	channel_array = np.array(img)
	grid_size_w = math.floor((N / gs))
	grid_size_h = math.floor((M / gs))
	cells = []


####################################################################################################
	c_index = 0
	for r in range(0,N - grid_size_w + 1, grid_size_w):
		for c in range(0,M - grid_size_h + 1, grid_size_h):
			c_sip = encodeinteger.encodeInteger(int(keys[c_index]))
			c_size = len(c_sip)
			print("Current embeded key: ",keys[c_index])
			grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]
			g_cell = Image.fromarray(grid_cell)
			w_im,s = em.getWatermarkedImage(g_cell,c_sip,c_size,k,IMAGE_NAME,COPT,pr,pb,im_format,root_path)
			cells.append(w_im)
			c_index += 1
#####################################################################################################
	img = mergeCellsToImage(cells,grid_size_w,grid_size_h,gs,im_format,s)
	img = cv2.resize(np.array(img),(w,h),interpolation = cv2.INTER_AREA)
	#img = Image.fromarray(img)
	#path = out_path + "/watermarked_" + IMAGE_NAME + "_1" + "." + im_format
	#img.save(path,quality = 100)

	return img
				

if __name__ == '__main__':
	path = sys.argv[1]
	key = int(sys.argv[2])
	sip = encodeinteger.encodeInteger(key)
	size = len(sip)
	im_name = sys.argv[3]
	c = float(sys.argv[4])
	img = recursive_embed(path,sip,size,im_name,c,2,2)