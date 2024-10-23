from datetime import datetime
from cryptography.fernet import Fernet
from PIL import Image
from codeMapping import *
from metrics import *
from initializer import *
from decrypter import *
import numpy as np,itertools as it, time, cv2, os, sys, subprocess
import logging



logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Creating a logger
logger = logging.getLogger(__name__)

def calculateBasicValues(M, N, size, PR, PB):
	gridSize,RBWidth,Rxy,Bxy,qSize,dSize = [],[],[],[],(4 * size),(2 * size)
	if(N < M):
		gridWidth = math.floor(N / dSize) - 6
		gridHeight = math.floor(N / dSize) - 6
		gridSize = [gridWidth, gridHeight]

		RED_WIDTH = PR
		BLUE_WIDTH = PB
		RBWidth = [PR, PB]

		RED_RADIOUS_X = math.floor(N / qSize)
		RED_RADIOUS_Y = math.floor(N / qSize)
		Rxy = [RED_RADIOUS_X,RED_RADIOUS_Y]

		BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
		BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		Bxy = [BLUE_RADIOUS_X,BLUE_RADIOUS_Y]
	elif(N > M):
		gridWidth = math.floor(M / dSize)
		gridHeight = math.floor(M / dSize)
		gridSize = [gridWidth, gridHeight]

		RED_WIDTH = PR
		BLUE_WIDTH = PB
		RBWidth = [PR, PB]

		RED_RADIOUS_X = math.floor(M / qSize)
		RED_RADIOUS_Y = math.floor(M / qSize)
		Rxy = [RED_RADIOUS_X,RED_RADIOUS_Y]

		BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
		BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		Bxy = [BLUE_RADIOUS_X,BLUE_RADIOUS_Y]
	else:
		gridWidth = math.floor(M / dSize)
		gridHeight = math.floor(N / dSize)
		gridSize = [gridWidth, gridHeight]

		RED_WIDTH = PR
		BLUE_WIDTH = PB
		RBWidth = [PR, PB]

		RED_RADIOUS_X = math.floor(M / qSize)
		RED_RADIOUS_Y = math.floor(N / qSize)
		Rxy = [RED_RADIOUS_X,RED_RADIOUS_Y]

		BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
		BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		Bxy = [BLUE_RADIOUS_X,BLUE_RADIOUS_Y]
	return gridSize,RBWidth,Rxy,Bxy

def mergeCellsToImage(cells, w, h, N) :
	display = np.empty(((h)*N, (w)*N , 3), dtype = np.uint8)
	for i, j in it.product(range(N), range(N)):
		arr = np.array(cells[i*N + j])
		x,y = i*(h), j*(w)
		display[x : x + (h), y : y + (w)] = arr
	return display

def writeBestCValuesInFile(optimalCValues, codeTaken, extractionRate, subpath):
	path = os.path.join(subpath, "BestCValues.txt")
	try :
		f = open(path,"w")
	except :
		print("File cannot be opened")
		exit(1)

	f.write("This file contains optimized c values for the original image.\n")
	f.write("Time and Date produced : " + str(datetime.now()) + "\n")

	for i in range(len(optimalCValues)):
		f.write("For Block " + str(i + 1) + " the optimized C value is " + str(optimalCValues[i]) + "\n")
	f.write("Code we took after final extraction = " + str(codeTaken) + "\n")
	f.write("Extraction percentage = " + str(extractionRate) + "%\n")
	f.close()

def getCellsFromAttacked(attackedImage, size):
	cells = []
	M,N = attackedImage.size
	channel_array = np.array(attackedImage)
	blockWidth,blockHeight = getBlockDimensions(M, N, size) 
	for r in range(0, (N - blockHeight + 1), blockHeight):
		for c in range(0, (M - blockWidth + 1), blockWidth):
			grid_cell = channel_array[r:r + blockHeight, c:c + blockWidth]
			g_cell = Image.fromarray(grid_cell)
			cells.append(g_cell)
	return cells

def getWatermarkedBlock(comCells, index, em, sip, compressedImageName, cbest, gridSize, RBWidth, Rxy, Bxy, moves):
	print("Running with c = " + str(cbest))
	g_cell = comCells[index]
	w_im = em.getWatermarkedImage(g_cell, sip, len(sip), compressedImageName, cbest, 2, 2, gridSize, RBWidth, Rxy, Bxy, moves)
	return g_cell, w_im

def saveWatermarkedImage(watermarkedImageName, watermarkedImage, dictionary, code, codeSips, flag):

	script_directory = "/storage/emulated/0/Android/data/com.vungn.camerax/files/Pictures/CameraXApp/"
	logger.debug(script_directory)
	watermarked_path = os.path.join(script_directory, "watermarked")
	logger.debug(watermarked_path)
	subpath = os.path.join(watermarked_path, watermarkedImageName)
	logger.debug(subpath)

	if not os.path.exists(watermarked_path):
		os.makedirs(watermarked_path)
	if not os.path.exists(subpath):
		os.makedirs(subpath)
	if os.path.exists(os.path.join(subpath, (watermarkedImageName + ".jpg"))):
		os.remove(os.path.join(subpath, (watermarkedImageName + ".jpg")))

	rotated_image = rotate_image_by_90(watermarkedImage)

	rotated_image.save((os.path.join(subpath, (watermarkedImageName + ".jpg"))), quality = 100)


	mapping_path = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/', ".Code_Mapping.txt")  # This will write to the user's home directory
	encryption_path = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/', ".encryption_key.txt")
	config_path = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/', ".config.txt")

	if (flag == 1) and (not(os.path.exists(config_path))):
		logger.debug(flag)
		try:
			mapping = open(mapping_path, "w")
			print(f"Successfully opened {mapping_path} for writing.")
		except IOError as e:
			print(f"File cannot be opened: {e}")
			print(f"Current working directory: {os.getcwd()}")
			print(f"Directory contents: {os.listdir('.')}")
			exit(1)

		mapping.write("This file contains the mapping for the chosen code.\n")
		mapping.write("Time and Date produced : " + str(datetime.now()) + "\n")
		mapping.write("Code given : " + str(code) + "\n")
		mapping.write("Sips for given code : " + str(codeSips) + "\n")
		for key in dictionary:
			mapping.write(str(key) + " : " + str(dictionary[key]) + "\n")
		mapping.close()

		password = b'4327'
		key = Fernet.generate_key()
		encrypt_file(mapping_path, key)
		os.remove(mapping_path)
		with open(encryption_path, 'wb') as file:
			file.write(key)
		#subprocess.run(["icacls", encryption_path, "/inheritance:r", "/grant:r", "Administrators:(F)"], check=True)
	return subpath,mapping_path

def calculateElapseTimeAndPrintResults(start, extractionRate):
	# Elapsed time for the algorithm
	print("Extraction percentage = " + str(extractionRate) + "%\n")
	end = time.time()
	secSTR,minSTR = calculateElapseTime(start, end)
	print("Elapsed time = " + str(minSTR) + " mins")
	print("Elapsed time = " + str(secSTR) + " seconds")

def calculateElapseTime(start, end):
	elapsedTimeInSeconds = end - start
	elapsedTimeInMinutes = elapsedTimeInSeconds / 60
	secSTR = "{:.3f}".format(elapsedTimeInSeconds)
	minSTR = "{:.3f}".format(elapsedTimeInMinutes)
	return secSTR,minSTR

def decodeKey(key1, key2, key3, innerKey):
	if(key1 == innerKey):
		return key1
	elif(key2 == innerKey):
		return key2
	elif(key3 == innerKey):
		return key3
	else:
		return "X"

def writeGridPlacementsInFile(allGridPositions):
	filename = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/', ".config.txt")
	encryption_path = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/', ".encryption_key.txt")
	logger.debug(encryption_path)
	logger.debug(filename)
	with open(filename, 'w') as file:
		for i in range(len(allGridPositions)):
			file.write(str(i + 1) + ": " + str(allGridPositions[i][0]) + "," + str(allGridPositions[i][1]) + "\n")
		file.close()
		#os.chmod(filename, 0o444)
	#subprocess.call(['attrib', '+H', filename])
	password = b'4327'
	key = Fernet.generate_key()
	encrypt_file(filename, key)
	logger.debug(encrypt_file(filename,key))
	with open(encryption_path, 'wb') as file:
		file.write(key)
	#subprocess.run(["icacls", encryption_path, "/inheritance:r", "/grant:r", "Administrators:(F)"], check=True)
	os.remove(filename)

def readGridlacementsFromFile():
	#subprocess.run(["icacls", "encryption_key_config.txt", "/reset"], check=True)
	encrypted_file_path = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/','.config.txt.encrypted')
	with open(os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/','.encryption_key.txt'), 'rb') as file:
		key = file.read()
	#subprocess.run(["icacls", 'encryption_key_config.txt', "/inheritance:r", "/grant:r", "Administrators:(F)"], check=True)
	decrypted_content = decrypt_file(encrypted_file_path, key)
	with open(os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/','.config.txt'), 'wb') as file:
		file.write(decrypted_content)
	#subprocess.call(['attrib', '+H', 'config.txt'])
	gridPositions = []
	try:
		with open(os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/','.config.txt'), 'r') as file:
			for i in range(0,36):
				content = file.readline()
				content = content.split(":")
				pos = content[1].strip()
				pos = pos.split(",")
				gridPosition = [int(pos[0]),int(pos[1])]
				gridPositions.append(gridPosition)
		os.remove(os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/','.config.txt'))
		return gridPositions
	except FileNotFoundError:
		print("Configuration file not found. Please create the file first.")
		sys.exit(1)

def encrypt_file(file_path, key):
    with open(file_path, 'rb') as file:
        data = file.read()

    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data)
    with open(file_path + ".encrypted", 'wb') as file:
        file.write(encrypted_data)

def checkInputFlagState(flag, filename):
	if((flag == 0) and (os.path.exists(filename))):
		allGridPositions = readGridlacementsFromFile()
		return allGridPositions
	elif((flag == 0) and (not (os.path.exists(filename)))):
		print("The config file is not present. Stopping....")
		sys.exit(1)
	else:
		if(os.path.exists(filename)):
			print("The config file is present. Stopping....")
			sys.exit(1)

def checkForFirstRun(flag, filename):
	if((flag == 1) and (not (os.path.exists(filename)))):
		return True
	else:
		return False


def rotate_image_by_90(image):
	# Rotate the image by 90 degrees
	return image.rotate(-90, expand=True)