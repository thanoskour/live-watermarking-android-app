import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import dft
import ellipticdisk
#from .ellipticdisk import createEllipticDisk
from math import sqrt
from PIL import Image
import itertools as it
import threading, queue
import math

class ExtractPermutation:
	def __init__(self):
		return
	def openImage(self,path):
		img = Image.open(path)
		return img

	def getFFTTransform(self,image):
		dft = np.fft.fft2(image,norm='ortho')
		fftShift = np.fft.fftshift(dft)
		mag = np.abs(fftShift)
		phase = np.angle(fftShift)
		return mag,phase

	def getIFFTTransform(self,mag,phase):
		real = mag * np.cos(phase)
		imag = mag * np.sin(phase)
		complex_output = np.zeros(mag.shape, complex)

		complex_output.real = real
		complex_output.imag = imag
		back_ishift = np.fft.ifftshift(complex_output)
		img_back = np.fft.ifft2(back_ishift,norm='ortho')
		img_back = abs(img_back)
		return img_back 

	def showImage(self,*image_args):
		for image in image_args:
			cv2.imshow('image',image)
			cv2.waitKey(0)
		cv2.waitKey(0)
		return

	def getSip(self,img,SIZE,K,PR,PB,im_format):
		#img = Image.open(img)
		im1 = np.array(img)
		M,N = img.size

		if(N == M):
			if(im_format == 'jpg'):
				r,g,b = img.split()
			else:
				r = Image.fromarray(im1[:,:,0])
				g = Image.fromarray(im1[:,:,1])
				b = Image.fromarray(im1[:,:,2])

			sip1 = self.extractPermutationFromChannel(r,SIZE,K,PR,PB)
			sip2 = self.extractPermutationFromChannel(g,SIZE,K,PR,PB)
			sip3 = self.extractPermutationFromChannel(b,SIZE,K,PR,PB)
		else:
			if(im_format == 'jpg'):
				r,g,b = img.split()
			else:
				r = Image.fromarray(im1[:,:,0])
				g = Image.fromarray(im1[:,:,1])
				b = Image.fromarray(im1[:,:,2])

			sip1 = self.extractPermutationFromChannel(r,SIZE,K,PR,PB)
			sip2 = self.extractPermutationFromChannel(g,SIZE,K,PR,PB)
			sip3 = self.extractPermutationFromChannel(b,SIZE,K,PR,PB)

		return sip1,sip2,sip3

	def extractPermutationFromChannel(self,channel,SIZE,K,PR,PB):

		grid_cell_num = 0
		sip_cells = []
		sip = []
		avg = []
		# STEP 2: COMPUTE IMAGE SIZE

		M,N = channel.size
		channel_array = np.array(channel)
		#print(N,M)

		#print("grid: ",K)
		#channel_array = cv2.resize(channel_array,(200,200))

		# GET THE SIZE OF EACH GRID SHELL

		if(N < M):
			grid_size_w = math.floor(((N) / SIZE))
			grid_size_h = math.floor(((N) / SIZE))
			#print(grid_size_w,grid_size_h)
			RED_WIDTH = PR
			BLUE_WIDTH = PB

			RED_RADIOUS_X = math.floor(((N) / (2 * SIZE)))
			RED_RADIOUS_Y = math.floor(((N) / (2 * SIZE)))

			BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
			BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		elif(N > M):
			grid_size_w = math.floor(((M) / SIZE))
			grid_size_h = math.floor(((M) / SIZE))
			#print(grid_size_w,grid_size_h)
			RED_WIDTH = PR
			BLUE_WIDTH = PB

			RED_RADIOUS_X = math.floor(((M) / (2 * SIZE)))
			RED_RADIOUS_Y = math.floor(((M) / (2 * SIZE)))

			BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
			BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		else:
			grid_size_w = math.floor(((M) / SIZE))
			grid_size_h = math.floor(((N) / SIZE))
			#print(grid_size_w,grid_size_h)
			RED_WIDTH = PR
			BLUE_WIDTH = PB

			RED_RADIOUS_X = math.floor(((M) / (2 * SIZE)))
			RED_RADIOUS_Y = math.floor(((N) / (2 * SIZE)))

			BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
			BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)

		minAvg = []
		minAll = []

		x = 0
		y = 0
		i = 0
		c = 0
		mag_red_blue = []
		mag_rest = []

		if(N < M):
			for r in range(0,(N) - grid_size_h + 1, grid_size_h):
				for c in range(K,(K+N) - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					
					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

					mag,phase = self.getFFTTransform(grid_cell)
					
					#print(grid_cell.shape)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = ellipticdisk.createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = ellipticdisk.createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					#print(AVG_RED)
					AVG_BLUE = sum(blue) / len(blue)
					#print(AVG_BLUE)
					avg.append(AVG_RED)

					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					#print("local min ",minAvg)

					y += 1
				minAll.append(min(minAvg))
				#print("min all ",minAll)
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip
		
		elif(N > M):

			for r in range(K,(M+K) - grid_size_h + 1, grid_size_h):
				for c in range(0,(M) - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					
					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

					mag,phase = self.getFFTTransform(grid_cell)
					
					#print(grid_cell.shape)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = ellipticdisk.createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = ellipticdisk.createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					#print(AVG_RED)
					AVG_BLUE = sum(blue) / len(blue)
					#print(AVG_BLUE)
					avg.append(AVG_RED)

					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					#print("local min ",minAvg)

					y += 1
				minAll.append(min(minAvg))
				#print("min all ",minAll)
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip
		else:
		
			for r in range(0,N - grid_size_w + 1, grid_size_w):
				for c in range(0,M - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					
					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

					mag,phase = self.getFFTTransform(grid_cell)
					
					#print(grid_cell.shape)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = ellipticdisk.createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = ellipticdisk.createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					#print(AVG_RED)
					AVG_BLUE = sum(blue) / len(blue)
					#print(AVG_BLUE)
					avg.append(AVG_RED)

					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					#print("local min ",minAvg)

					y += 1
				minAll.append(min(minAvg))
				#print("min all ",minAll)
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip