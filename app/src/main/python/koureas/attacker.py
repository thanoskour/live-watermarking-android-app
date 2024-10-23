import numpy as np
from PIL import Image
import cv2

# Bluring filters: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

def attackWatermarkedImage(watermarkedImage, mode):
	if(mode == "Compression"):
		return compressWatermarkedImage(watermarkedImage)
	elif(mode == "Gaussian Blur"):
		return gaussianBlurWatermarkedImage(watermarkedImage)
	elif(mode == "Gaussian Noise"):
		return gaussianNoiseWatermarkedImage(watermarkedImage)
	elif(mode == "Salt and Pepper"):
		return saltAndPepperWatermarkedImage(watermarkedImage)
	elif(mode == "Histogram Equalization"):
		return histogramEQWatermarkedImage(watermarkedImage)
	else:
		return gammaWatermarkedImage(watermarkedImage)

def attackWatermarkedImageWithCrops(watermarkedImage, direction, percentage):
	wImage = np.array(Image.open(watermarkedImage))
	w,h = wImage.shape[1],wImage.shape[0]
	if(direction == "Left"):
		if(percentage == 25):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.25 * w)
			wImage[:,0:d] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		elif(percentage == 75):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.75 * w)
			wImage[:,0:d] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		else:
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.5 * w)
			wImage[:,0:d] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
	elif(direction == "Right"):
		if(percentage == 25):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.25 * w)
			wImage[:,(w - d):] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		elif(percentage == 75):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.75 * w)
			wImage[:,(w - d):] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		else:
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.5 * w)
			wImage[:,(w - d):] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
	elif(direction == "Top"):
		if(percentage == 25):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.25 * h)
			wImage[0:d,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		elif(percentage == 75):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.75 * h)
			wImage[0:d,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		else:
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.5 * h)
			wImage[0:d,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
	elif(direction == "Bottom"):
		if(percentage == 25):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.25 * h)
			wImage[(h-d):,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		elif(percentage == 75):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.75 * h)
			wImage[(h-d):,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		else:
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.5 * h)
			wImage[(h-d):,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
	elif(direction == "BothVertical"):
		if(percentage == 25):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.125 * w)
			wImage[:,0:d] = 0
			wImage[:,(w-d):] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		elif(percentage == 75):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.375 * w)
			wImage[:,0:d] = 0
			wImage[:,(w-d):] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		else:
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.25 * w)
			wImage[:,0:d] = 0
			wImage[:,(w-d):] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
	else:
		if(percentage == 25):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.125 * h)
			wImage[0:d,:] = 0
			wImage[(h-d):,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		elif(percentage == 75):
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.375 * h)
			wImage[0:d,:] = 0
			wImage[(h-d):,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName
		else:
			croppedImageName = "Croped" + direction + str(percentage) + watermarkedImage
			d = int(0.25 * h)
			wImage[0:d,:] = 0
			wImage[(h-d):,:] = 0
			croppedImage = Image.fromarray(wImage)
			croppedImage.save(croppedImageName, quality = 100)
			croppedImage = Image.open(croppedImageName)
			return croppedImage,croppedImageName

def compressWatermarkedImage(watermarkedImage):
	wImage = Image.open(watermarkedImage)
	compressedImageName = "Comp" + watermarkedImage
	wImage.save(compressedImageName, format = "JPEG", quality = 65, subsampling = 1)
	compressedImage = Image.open(compressedImageName)
	return compressedImage,compressedImageName

def gaussianNoiseWatermarkedImage(watermarkedImage):
	std = 15.5
	gnName = "GN" + watermarkedImage
	wImage = np.array(Image.open(watermarkedImage))
	w,h = wImage.shape[0],wImage.shape[1]
	gauss_noise = np.zeros((w,h,3), dtype = np.uint8)
	cv2.randn(gauss_noise, 0, std)
	gauss_noise = (gauss_noise*0.5).astype(np.uint8)
	noisy = cv2.add(wImage, gauss_noise)
	noisy_img = Image.fromarray(noisy)
	noisy_img.save(gnName, quality = 100, subsampling = 1)
	gnImage = Image.open(gnName)
	return gnImage,gnName

def saltAndPepperWatermarkedImage(watermarkedImage):
	sapName = "SAP" + watermarkedImage
	wImage = np.array(Image.open(watermarkedImage))
	w,h = wImage.shape[0],wImage.shape[1]
	for i in range(w):
		for j in range(h):
			random_num_1 = np.random.uniform(low = 0.0, high = 1.0)
			random_num_2 = np.random.uniform(low = 0.0, high = 1.0)
			if(random_num_1 < 0.05):
				wImage[i,j] = 255
			elif(random_num_2 < 0.05):
				wImage[i,j] = 0
			else:
				pass
	sap = Image.fromarray(wImage)
	sap.save(sapName, quality = 100, subsampling = 1)
	sapImage = Image.open(sapName)
	return sapImage,sapName

def gaussianBlurWatermarkedImage(watermarkedImage):
	gbName = "GB" + watermarkedImage
	wImage = np.array(Image.open(watermarkedImage))
	imblur = cv2.GaussianBlur(wImage, (5, 5), 0)
	imblur = Image.fromarray(imblur)
	imblur.save(gbName, quality = 100, subsampling = 1)
	gbImage = Image.open(gbName)
	return gbImage,gbName

def histogramEQWatermarkedImage(watermarkedImage):
	heqName = "HEQ" + watermarkedImage
	wImage = np.array(Image.open(watermarkedImage))
	img = cv2.cvtColor(wImage, cv2.COLOR_BGR2YCrCb)
	y, cr, cb = cv2.split(img)
	y_eq = cv2.equalizeHist(y)
	img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
	img_heq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
	img_heq = Image.fromarray(img_heq)
	img_heq.save(heqName, quality = 100, subsampling = 1)
	heqImage = Image.open(heqName)
	return heqImage,heqName

def gammaWatermarkedImage(watermarkedImage):
	gammaName = "Gamma" + watermarkedImage
	wImage = np.array(Image.open(watermarkedImage))
	kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
	image_gamma = cv2.filter2D(wImage, -1, (1.625))
	image_gamma = Image.fromarray(image_gamma)
	image_gamma.save(gammaName, quality = 100, subsampling = 1)
	gammaImage = Image.open(gammaName)
	return gammaImage,gammaName