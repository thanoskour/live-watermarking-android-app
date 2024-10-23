import numpy as np
from PIL import Image
import math
import itertools as it


def create_rgb_image_from_channels(blocks,ch1,ch2,size,op,s):

	N,M = ch1.shape[0],ch1.shape[1]

	grid_size_w,grid_size_h = math.floor((N / size)), math.floor((M / size))

	dis = np.empty((N,M), dtype = np.uint8)
	if(s[2] == 3):
		im = np.empty((N,M,3), dtype = np.uint8)
		for i, j in it.product(range(size), range(size)):
			arr1 = blocks[i*size+j]
			x,y = i*(grid_size_w), j*(grid_size_h)
			dis[x : x + (grid_size_w), y : y + (grid_size_h)] = arr1

		if(op == 'red'):
			im[:,:,0] = dis
			im[:,:,1] = ch1
			im[:,:,2] = ch2
		elif(op == 'green'):
			im[:,:,0] = ch1
			im[:,:,1] = dis
			im[:,:,2] = ch2
		else:
			im[:,:,0] = ch1
			im[:,:,1] = ch2
			im[:,:,2] = dis
	else:
		im = np.empty((N,M,4), dtype = np.uint8)
		for i, j in it.product(range(size), range(size)):
			arr1 = blocks[i*size+j]
			x,y = i*(grid_size_w), j*(grid_size_h)
			dis[x : x + (grid_size_w), y : y + (grid_size_h)] = arr1

		if(op == 'red'):
			im[:,:,0] = dis
			im[:,:,1] = ch1
			im[:,:,2] = ch2
		elif(op == 'green'):
			im[:,:,0] = ch1
			im[:,:,1] = dis
			im[:,:,2] = ch2
		else:
			im[:,:,0] = ch1
			im[:,:,1] = ch2
			im[:,:,2] = dis

	return im


def create_gray_channel(blocks,ch,size):

	N,M = ch.shape[0],ch.shape[1]

	grid_size_w,grid_size_h = math.floor((N / size)), math.floor((M / size))

	dis = np.empty((N,M), dtype = np.uint8)

	for i, j in it.product(range(size), range(size)):
		arr1 = blocks[i*size+j]
		x,y = i*(grid_size_w), j*(grid_size_h)
		dis[x : x + (grid_size_w), y : y + (grid_size_h)] = arr1

	return dis


def produce_blocks(N,M,size,rch,gch,bch):

	g_block_ex = []
	r_block_ex = []
	b_block_ex = []

	N = rch.shape[0]
	M = rch.shape[1]

	grid_size_w = math.floor((N / size))
	grid_size_h = math.floor((M / size))

	for w in range(0,N - grid_size_w + 1, grid_size_w):
		for e in range(0,M - grid_size_h + 1, grid_size_h):
			grid_cell_g_ex = gch[w:w + grid_size_w, e:e + grid_size_h]
			grid_cell_r_ex = rch[w:w + grid_size_w, e:e + grid_size_h]
			grid_cell_b_ex = bch[w:w + grid_size_w, e:e + grid_size_h]

			g_block_ex.append(grid_cell_g_ex)
			r_block_ex.append(grid_cell_r_ex)
			b_block_ex.append(grid_cell_b_ex)

	return r_block_ex,g_block_ex,b_block_ex
