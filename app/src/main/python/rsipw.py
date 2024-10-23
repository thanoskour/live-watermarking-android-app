import openpyxl,cv2,sys,re,time,random,numpy as np,matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from openpyxl.utils import get_column_letter
import decodesip
#from decodesip import decodeSip
from math import log10, sqrt
from recossip import recsip
from PIL import Image
from initializer import *
from codeMapping import *
#from utilities import *
import utilities
import optimizer
#from optimizer import *
from attacker import *
from metrics import *
import encodeinteger
import logging
import os


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Creating a logger
logger = logging.getLogger(__name__)

PR = 2
PB = 2
FULL_PERCENTAGE = 100
NUM_OF_WATERMARKS = 36
FAST_MODE = "FAST"
FULL_MODE = "FULL"

optimalGridPositionForEachBlock,gridSize,RBWidth,Rxy,Bxy,optimalCValues = [],[],[],[],[],[]

def getCodeFromWatermarkedImage(code, watermarkedBlocks, codeSips, mapping, innerSips, allGridPositions, gridParams):
	logger.debug("getCodeFromWatermarkedImage: Start")
	gridSize,RBWidth,Rxy,Bxy = gridParams[0],gridParams[1],gridParams[2],gridParams[3]
	codeTaken,totalWatermarksExtracted = [],36
	for i in range(len(watermarkedBlocks)):
		watermarkedBlock,sip,bestMove,enable = watermarkedBlocks[i],codeSips[i],allGridPositions[i],1
		isExtracted,key1,key2,key3 = extractRecursiveSip(watermarkedBlock, IMAGE_PATH, sip, innerSips[i], code, gridSize, RBWidth, Rxy, Bxy, bestMove)
		decodedKey = utilities.decodeKey(key1, key2, key3, sip)
		if(decodedKey == "X"):
			enable = 0
			codeTaken.append("X")
			totalWatermarksExtracted = totalWatermarksExtracted - 1
		else:
			codeTaken.append(mapping[decodedKey])
		printResults(3, i, enable, 0, [])
	extractionRate = (totalWatermarksExtracted / NUM_OF_WATERMARKS)*FULL_PERCENTAGE
	logger.debug("getCodeFromWatermarkedImage: End")
	return codeTaken,extractionRate

def recursiveEmbed(code, mode, flag, inputPin):
	logger.debug("recursiveEmbed: Start")
	filename = os.path.join('/storage/emulated/0/Android/data/com.vungn.camerax/files/',".config.txt.encrypted")
	allGridPositions = utilities.checkInputFlagState(flag, filename)

	watermarkedBlocks,innerSips,index = [],[],0
	em,ex = init()
	img = openImage(IMAGE_PATH)
	M,N = img.size
	channel_array = np.array(img)

	#code = checkForFullCode(code)
	code = code + inputPin
	mappingDictionary = getCodeMapping(code)
	codeSips = getSipsFromCode(mappingDictionary, code)
	size = sqrt(len(codeSips))
	blockWidth,blockHeight = getBlockDimensions(M, N, size)

	for offsetY in range(0, (N - blockHeight + 1), blockHeight):
		for offsetX in range(0, (M - blockWidth + 1), blockWidth):
			innerKey = codeSips[index]
			logger.debug(f"Embed key : {innerKey} in Block {index + 1}")
			print("Embed key : " + str(innerKey) + " in Block " + str(index + 1))
			innerSip = encodeinteger.encodeInteger(innerKey)
			innerSips.append(innerSip)
			blockParams = [innerSip, len(innerSip), innerKey]
			grid_cell = channel_array[offsetY:(offsetY + blockHeight), offsetX:(offsetX + blockWidth)]
			g_cell = Image.fromarray(grid_cell)
			optimalCValue,watermarkedBlock,gridParams = findBestCValues(blockParams, blockWidth, blockHeight, em, code, g_cell, mode, 1, flag, index, allGridPositions)
			optimalCValues.append(optimalCValue)
			watermarkedBlocks.append(watermarkedBlock)
			index = index + 1
	watermarkedImage = utilities.mergeCellsToImage(watermarkedBlocks, blockWidth, blockHeight, int(size))
	watermarkedImage = Image.fromarray((cv2.resize(np.array(watermarkedImage), (M,N), interpolation = cv2.INTER_AREA)))
	wimgName = "watermarked_" + IMAGE_NAME
	subpath,mappingPath = utilities.saveWatermarkedImage(wimgName, watermarkedImage, mappingDictionary, code, codeSips, flag)
	if(utilities.checkForFirstRun(flag, filename)):
		utilities.writeGridPlacementsInFile(optimalGridPositionForEachBlock)
		allGridPositions = optimalGridPositionForEachBlock
	psnr,ssim = optimizer.getPSNRAndSSIM(np.array(watermarkedImage), channel_array)
	logger.debug("recursiveEmbed: End")

	return watermarkedBlocks,codeSips,mappingDictionary,innerSips,subpath,optimalCValues,allGridPositions,gridParams

def extractRecursiveSip(watermarkedBlock, imagePath, originalKey, innerSip, code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock):
	logger.debug("extractRecursiveSip: Start")
	em,ex = init()
	img = openImage(imagePath)
	M,N = img.size
	channel_array = np.array(img)

	sip1,sip2,sip3 = ex.getSip(watermarkedBlock, len(innerSip), PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	logger.debug(f"Key which was embeded : {originalKey}")
	print("Key which was embeded :", originalKey)
	key1 = decodesip.decodeSip(sip1)
	key2 = decodesip.decodeSip(sip2)
	key3 = decodesip.decodeSip(sip3)
	if(key1 == originalKey or key2 == originalKey or key3 == originalKey) :
		logger.debug("extractRecursiveSip: End with key match")
		return 1,key1,key2,key3
	logger.debug("extractRecursiveSip: End without key match")
	return 0,key1,key2,key3

def findBestCValues(blockParams, blockWidth, blockHeight, embedObject, code, g_cell, mode, extractionIsPrioritized, flag, index, allGridPositions) :
	logger.debug("findBestCValues: Start")

	gridSize,RBWidth,Rxy,Bxy = utilities.calculateBasicValues(blockWidth, blockHeight, blockParams[1], PR, PB)
	gridParams = [gridSize, RBWidth, Rxy, Bxy]
	if(mode == FAST_MODE):
		if(flag == 1):
			randomGridPosition = optimizer.calculateRandomGridPosition(blockParams, blockWidth, blockHeight, gridSize)
			optimalGridPositionForEachBlock.append(randomGridPosition) 
			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizer.optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, IMAGE_NAME, IMAGE_PATH, randomGridPosition)
			logger.debug("findBestCValues: End (fast mode, flag=1)")
			return optimalCValue,watermarkedBlock,gridParams
		else:
			gridPosition = allGridPositions[index]
			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizer.optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, IMAGE_NAME, IMAGE_PATH, gridPosition)
			logger.debug("findBestCValues: End (fast mode, flag=0)")
			return optimalCValue,watermarkedBlock,gridParams
	else:
		optimalCValue,watermarkedBlock,optimalGridPosition = optimizer.optimizeCValueFull(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, IMAGE_NAME, IMAGE_PATH, blockWidth, blockHeight)
		optimalGridPositionForEachBlock.append(optimalGridPosition)
		logger.debug("findBestCValues: End (full mode)")
		return optimalCValue,watermarkedBlock,gridParams



def main(image_path, code, flag, input_pin):
	global IMAGE_PATH, IMAGE_NAME
	IMAGE_PATH = image_path
	IMAGE_NAME = (((IMAGE_PATH.split("/"))[-1]).split(".jpg"))[0]

	start = time.time()
	logger.info("Processing started")
	code = getListFromCode(code)
	input_pin = getListFromCode(input_pin)

	# Run Main Algorithm
	watermarkedBlocks, codeSips, codeMapping, innerSips, subpath, optimalCValues, allGridPositions, gridParams = recursiveEmbed(code, FAST_MODE, flag, input_pin)
	logger.info("Processing completed")
	end = time.time()
	elapsed_time = end - start
	logger.info(f"Total processing time: {elapsed_time} seconds")
	# Return results or print them if needed
	return {"watermarkedBlocks": watermarkedBlocks, "codeSips": codeSips,"codeMapping":codeMapping,"innerSips": innerSips ,"subpath": subpath, "optimalCValues": optimalCValues, "allGridPositions": allGridPositions, "gridParams":gridParams}

if __name__ == "__main__":
	image_path = sys.argv[1]
	code = sys.argv[2]
	flag = int(sys.argv[3])
	input_pin = sys.argv[4]
	main(image_path, code, flag, input_pin)