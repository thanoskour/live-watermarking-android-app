import cv2
import numpy as np
from numpy import asarray
from PIL import Image
import math
import itertools as it
import hashlib
import matplotlib.pyplot as plt
import sys
import ast
import time
import hmac
import encodeinteger
import decodesip
import embed_key
import extract_sip
import recossip
import hash_ops
import image_helpers
#from .encodeinteger import encodeInteger
#from .decodesip import decodeSip
#from .embed_key import EmbedPermutation
#from .extract_sip import ExtractPermutation
#from .recossip import recsip
#from .hash_ops import hash_to_bits,xor_hashes,combine_hashes_and_hash,bits_to_hash
#from .image_helpers import create_rgb_image_from_channels,create_gray_channel,produce_blocks

class HashTD:
	def __init__(self,maph):
		self.MAP = maph
		return

	def common_divisors(self,dim):
		w = dim[0]
		h = dim[1]
		cm = []
		for i in range(1,min(w,h)+1):
			if((w % i) == (h % i) == 0):
				cm.append(i)
		return cm

	def hash_block(self,b):
		h = hashlib.new('sha256')
		h.update(b.tobytes())
		hashb = h.hexdigest()
		return hashb

	def store_private_key(self,pk):
		f = open("Data/private_key.txt","w+")
		f.write(pk)


	def embed_hash(self,hash_value,block,step):
		mod_block = []
		block = block.flatten()
		bits = hash_ops.hash_to_bits(hash_value)
		index = 0
		if(step == 0):
			for i in range(0,len(block)):
				sample = block[i]
				bits_sample = list(''.join(format(ord(chr(sample)), '08b')))
				bits_sample[-1] = bits[index]
				new_sample = int(''.join(bits_sample),2)
				block[i] = new_sample
				index += 1
				
		else:
			for i in range(0,len(block),step):
				sample = block[i]
				bits_sample = list(''.join(format(ord(chr(sample)), '08b')))
				bits_sample[-1] = bits[index]
				new_sample = int(''.join(bits_sample),2)
				block[i] = new_sample
				index += 1
				if(index > 255):
					break

		return block

	def xor_sip_hashes(self,hashes):
		private_key = []

		hash_bits_arr_T = hash_ops.xor_hashes(hashes)
		
		for i in range(len(hash_bits_arr_T)):
			res = 0
			for j in range(6):
				res ^= int(hash_bits_arr_T[i][j])
			private_key.append(res)

		pk = [str(i) for i in private_key]
		h_pk = hash_ops.bits_to_hash(pk)

		return h_pk


	def sort_blocks(self,mod_blocks):
		sorted_b = []
		m = sorted(mod_blocks, key = lambda x: x[1])
		for i in range(len(m)):
			sorted_b.append(m[i][0])
		return sorted_b

		
	def extract_hash(self,block,step):
		block = block.flatten()
		lsb =[]
		index = 0
		if(step == 0):
			for i in range(0,len(block)):
				sample = block[i]
				bits_sample = list(''.join(format(ord(chr(block[i])), '08b')))
				lsb.append(bits_sample[-1])
				index += 1
				if(index > 255):
					break
		else:
			for i in range(0,len(block),step):
				sample = block[i]
				bits_sample = list(''.join(format(ord(chr(block[i])), '08b')))
				lsb.append(bits_sample[-1])
				index += 1
				if(index > 255):
					break

		hash_val = hash_ops.bits_to_hash(lsb)
		
		return hash_val





	def embed_hashes(self,im,name,size,outpath,root_path):
		
		s = im.shape
		print("Input shape: ",s) 

		
		# convert each channel to numpy array
		if(s[2] == 3):
			r_ch = im[:,:,0]
			g_ch = im[:,:,1]
			b_ch = im[:,:,2]
		else:
			r_ch = im[:,:,0]
			g_ch = im[:,:,1]
			b_ch = im[:,:,2]
			a_ch = im[:,:,3]


		print("Input channel shape: ",b_ch.shape)
		CD = self.common_divisors(b_ch.shape)
		print("Common Divisors of ",b_ch.shape[0]," and ",b_ch.shape[1],": ",CD)
		#print(r_ch.shape,b_ch.shape,g_ch.shape)

		hashes_r = []
		mod_blocks = []

		N = r_ch.shape[0]
		M = r_ch.shape[1]


		grid_size_w = math.floor((N / size))
		grid_size_h = math.floor((M / size))

		step = (math.floor((grid_size_w * grid_size_h) / 256)) - 1
	
		print("Block sample step: ",step)
		print("Total image blocks: ",(size * size))
		print("Block shape: ",(grid_size_w,grid_size_h))
		
		c = 0
		h_f = open(root_path + "/hashes.txt","w+")
		
		# Produce 3 lists with the corresponding RGB blocks
		r_block,g_block,b_block = image_helpers.produce_blocks(N,M,size,r_ch,g_ch,b_ch)

		for i in range(len(r_block)):
			h_value = self.hash_block(r_block[i])
			hashes_r.append(h_value)
			h_f.write(str(c) + ":" + h_value + "\n")


		print("Total hashes R: ",len(hashes_r))
		


		f = open(root_path + "/map.txt","w+")
		for i in range(len(g_block)):
			green_block =  self.MAP[i][1]
			red_block = self.MAP[i][0]
			f.write("(" + str(red_block) + "," + str(green_block) + ")" + "\n")
			modified_block = self.embed_hash(hashes_r[red_block],g_block[green_block],step)
			#print(modified_block.shape)
			modified_block = np.reshape(modified_block,(grid_size_w,grid_size_h))
			mod_blocks.append((modified_block,green_block))
		
		new_mod_blocks = self.sort_blocks(mod_blocks)

		if(s[2] == 3):
			im = image_helpers.create_rgb_image_from_channels(new_mod_blocks,r_ch,b_ch,size,'green',s)
		else:
			im = image_helpers.create_rgb_image_from_channels(new_mod_blocks,r_ch,b_ch,size,'green',s)
			im[:,:,3] = a_ch

		print("Output image shape: ",im.shape)
		print("EC: ",(size*size)*256," bits")
		ec = (size*size)*256

		rgb = Image.fromarray(im)
		path = outpath + "/romfaia_" + name + "_s" + str(size) + ".png"
		rgb.save(path)
		h_f.close()
		f.close()

		return path


	###########################################################################
	# EXTRACTION PART FROM LINE 350
	###########################################################################

	# First time using lambda function coollll
	compare_hashes = lambda self,h1,h2: h1 == h2

	# Back channel propagation fixes FP hashes
	# Designed by Joe Polenakis
	def back_channel_propagation(self,r_block_ex,ex_hashes,maph,root_path):
		pairs = []
		false_blocks = []
		tampered_blocks = []
		FP = []
		FP_F = []
		em_hashes = []
		f1 = open(root_path + "/extracted_r_hashes.txt","w+")

		for i in range(len(r_block_ex)):
			h = self.hash_block(r_block_ex[i])
			em_hashes.append(h)
			f1.write( str(i) + ": " + h + "\n")

		for i in range(len(maph)):
			b = maph[i][0]
			c = maph[i][1]
			hb = em_hashes[b]
			hc = ex_hashes[c]
			if(self.compare_hashes(hb,hc)):
				continue
			else:
				#----- 1st choice ------------------
				for i in range(len(maph)):
					if(b == maph[i][1]):
						a = em_hashes[maph[i][0]]
						b_ = ex_hashes[maph[i][1]]
				if(self.compare_hashes(a,b_)):
					continue
				else:
					tampered_blocks.append(b)

				#-------- 2st choice----------------
				for i in range(len(maph)):
					if(c == maph[i][0]):
						c_ = em_hashes[maph[i][0]]
						d = ex_hashes[maph[i][1]]
				if(self.compare_hashes(c_,d)):
					continue
				else:
					tampered_blocks.append(c)

		print("Tampered blocks fixed: ",tampered_blocks," ",len(tampered_blocks))

		return tampered_blocks

	def extract_hashes(self,w_im,name,size,maph,root_path):

		f = open(root_path + "/data/extracted_hashes.txt","w+")

		s = (np.array(Image.open(w_im))).shape
		print(s)

		im1 = np.array(Image.open(w_im))

		if(s[2] == 3):
			r_ch = im1[:,:,0]
			b_ch = im1[:,:,2]
			g_ch = im1[:,:,1]
		else:
			r_ch = im1[:,:,0]
			b_ch = im1[:,:,2]
			g_ch = im1[:,:,1]
			a_ch = im1[:,:,3]

		N = g_ch.shape[0]
		M = g_ch.shape[1]

		grid_size_w = math.floor((N / size))
		grid_size_h = math.floor((M / size))


		step = (math.floor((grid_size_w * grid_size_h) / 256)) - 1

		g_block_ex = []
		r_block_ex = []
		b_block_ex = []

		em_hashes = []
		ex_hashes = []

		for w in range(0,N - grid_size_w + 1, grid_size_w):
			for e in range(0,M - grid_size_h + 1, grid_size_h):
				grid_cell_g_ex = g_ch[w:w + grid_size_w, e:e + grid_size_h]
				grid_cell_r_ex = r_ch[w:w + grid_size_w, e:e + grid_size_h]
				grid_cell_b_ex = b_ch[w:w + grid_size_w, e:e + grid_size_h]

				g_block_ex.append(grid_cell_g_ex)
				r_block_ex.append(grid_cell_r_ex)
				b_block_ex.append(grid_cell_b_ex)

		for i in range(len(g_block_ex)):
			try:
				hash_val = self.extract_hash(g_block_ex[i],step)
				ex_hashes.append(hash_val)
				f.write(str(i) + ":" + hash_val + "\n")
			except:
				f.write("Hash from Green Block " + str(i) + ":" + str(0) + "\n")


		f.close()

		fixed_list = self.back_channel_propagation(r_block_ex,ex_hashes,maph,root_path)

		
		for i in range(len(fixed_list)):
			r_block_ex[fixed_list[i]] = 255

		disg = image_helpers.create_gray_channel(g_block_ex,g_ch,size)
		disb = image_helpers.create_gray_channel(b_block_ex,b_ch,size)
		if(s[2] == 3):
			im = image_helpers.create_rgb_image_from_channels(r_block_ex,disg,disb,size,'red',s)
		else:
			im = image_helpers.create_rgb_image_from_channels(r_block_ex,disg,disb,size,'red',s)
			im[:,:,3] = a_ch

		out = Image.fromarray(im)
		out.save(name + "_detected.png")

		if(len(fixed_list) == 0):
			return "Image is valid"
		else:
			return "Image is not valid"
		
		

def create_map(size,option,root_path):
	MAP = []
	if(size == 8):
		f = open(root_path + "/permutations_out_8.txt","r")
		Lines1 = f.readlines()
		for line1 in Lines1:
			vals = line1.strip("\n").split(",")
			if(len(vals) == 0):
				continue
			else:
				a = int(vals[0])
				b = int(vals[1])
				MAP.append((a,b))
	if(size == 16):
		if(option == 'linear'):
			f = open(root_path + "/permutations_out_256_b.txt","r")
			Lines1 = f.readlines()
			for line1 in Lines1:
				vals = line1.strip("\n").split(",")
				if(len(vals) == 0):
					continue
				else:
					a = int(vals[0])
					b = int(vals[1])
					MAP.append((a,b))
		else:
			f = open(root_path + "/permutations_out_256.txt","r")
			Lines1 = f.readlines()
			for line1 in Lines1:
				vals = line1.strip("\n").split(",")
				if(len(vals) == 0):
					continue
				else:
					a = int(vals[0])
					b = int(vals[1])
					MAP.append((a,b))
	if(size == 32):
		if(option == 'linear'):
			f = open(root_path + "/permutations_out_1024_c.txt","r")
			Lines1 = f.readlines()
			for line1 in Lines1:
				vals = line1.strip("\n").split(",")
				if(len(vals) == 0):
					continue
				else:
					a = int(vals[0])
					b = int(vals[1])
					MAP.append((a,b))
		else:
			f = open(root_path + "/permutations_out_1024.txt","r")
			Lines1 = f.readlines()
			for line1 in Lines1:
				vals = line1.strip("\n").split(",")
				if(len(vals) == 0):
					continue
				else:
					a = int(vals[0])
					b = int(vals[1])
					MAP.append((a,b))
	if(size == 64):
		f = open(root_path + "/permutations_out_4096.txt","r")
		Lines1 = f.readlines()
		for line1 in Lines1:
			vals = line1.strip("\n").split(",")
			if(len(vals) == 0):
				continue
			else:
				a = int(vals[0])
				b = int(vals[1])
				MAP.append((a,b))
	if(size == 60):
		f = open(root_path + "/permutations_out_3600.txt","r")
		Lines1 = f.readlines()
		for line1 in Lines1:
			vals = line1.strip("\n").split(",")
			if(len(vals) == 0):
				continue
			else:
				a = int(vals[0])
				b = int(vals[1])
				MAP.append((a,b))

	if(size == 128):
		f = open(root_path + "/permutations_out_16384.txt","r")
		Lines1 = f.readlines()
		for line1 in Lines1:
			vals = line1.strip("\n").split(",")
			if(len(vals) == 0):
				continue
			else:
				a = int(vals[0])
				b = int(vals[1])
				MAP.append((a,b))
	return MAP

if __name__ == '__main__':
	arg1 = sys.argv[1]
	image = sys.argv[2]
	name = sys.argv[3]
	size = int(sys.argv[4])
	option = sys.argv[5]

	MAP = create_map(size,option)
	model = HashTD(MAP)
	
	if(arg1 == "embed"):
		print("Starting Tamper Protection Algorithm ...")
		start_time = time.time()
		hr,cap,im = model.embed_hashes(image,name,size)
		print("Total time to run embeding: ",time.time() - start_time,"secs")
		info = open("Data/proscess_info.txt","w+")
		info.write("Total hashes: " + str(len(hr)) + "\n")
		info.write("EC: " + str(cap) + "\n")
		info.write("Total time elapsed: " + str(time.time() - start_time) + " secs" + "\n")
	elif(arg1 == "detect"):
		start_time = time.time()
		im = model.extract_hashes(image,name,size,MAP)
		print("Total time to run embeding: ",time.time() - start_time,"secs")
		plt.imshow(im,cmap = 'gray',extent=[0,size,size,0])
		plt.show()
	else:
		pass


