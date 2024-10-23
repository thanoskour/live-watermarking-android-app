import sys
import re
import rsipw
#from .rsipw import recursive_embed
import time
import metrics
#from .metrics import PSNR,ssim_metric
import random
import hashlib
import os



def startEmbeding(path,keys,k,im_name,c,gs,id,root_path):

	_, img_format = os.path.splitext(path)
	img_format = img_format.lstrip('.').lower()
	print("Image format: ",img_format)

	im_name = im_name.split('.')[0]


	print("Image format:", img_format)
	print("Image name:", im_name)
	img = rsipw.recursive_embed(path,keys,k,im_name,c,2,2,gs,img_format,root_path)
	return img



if __name__ == '__main__':
	print("Repetitive embending started ...")
	
	keys = []


	path = sys.argv[1]
	c = float(sys.argv[2])
	gs = int(sys.argv[3])
	id = sys.argv[4]
	im_name = re.split(r'\.(?!\d)',path)[0]
	k = 140


	for i in range((gs**2)):
		val = random.randint(8, 15)
		keys.append(str(val))


	print("Inserted Keys: ",keys)
	start_time = time.time()
	img = startEmbeding(path,keys,k,im_name,c,gs,id,"w_images_4k")
	print("Elapsed time(sec): ","{:.2f}".format(time.time() - start_time))
	print("Repetitive embending ended ...")

	# ARGS:
	# 	arg1: image path
	# 	arg2: user's integer number (range 1-100)
	# 	arg3: output image name
	# 	arg4: float number for every block (range 0.01-100)
	# RETURN:
	# 	The program return the watermark image