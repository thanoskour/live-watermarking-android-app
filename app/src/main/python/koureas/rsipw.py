import openpyxl,cv2,sys,re,time,random,numpy as np,matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from openpyxl.utils import get_column_letter
from decodesip import decodeSip
from math import log10, sqrt
from recossip import recsip
from PIL import Image
from initializer import *
from codeMapping import *
from utilities import *
from optimizer import *
from attacker import *
from metrics import *

PR = 2
PB = 2
FULL_PERCENTAGE = 100
NUM_OF_WATERMARKS = 36
FAST_MODE = "FAST"
FULL_MODE = "FULL"

optimalGridPositionForEachBlock,gridSize,RBWidth,Rxy,Bxy,optimalCValues = [],[],[],[],[],[]

def getCodeFromWatermarkedImage(code, watermarkedBlocks, codeSips, mapping, innerSips, allGridPositions, gridParams):
	gridSize,RBWidth,Rxy,Bxy = gridParams[0],gridParams[1],gridParams[2],gridParams[3]
	codeTaken,totalWatermarksExtracted = [],36
	for i in range(len(watermarkedBlocks)):
		watermarkedBlock,sip,bestMove,enable = watermarkedBlocks[i],codeSips[i],allGridPositions[i],1
		isExtracted,key1,key2,key3 = extractRecursiveSip(watermarkedBlock, IMAGE_PATH, sip, innerSips[i], code, gridSize, RBWidth, Rxy, Bxy, bestMove)
		decodedKey = decodeKey(key1, key2, key3, sip)
		if(decodedKey == "X"):
			enable = 0
			codeTaken.append("X")
			totalWatermarksExtracted = totalWatermarksExtracted - 1
		else:
			codeTaken.append(mapping[decodedKey])
		printResults(3, i, enable, 0, [])
	extractionRate = (totalWatermarksExtracted / NUM_OF_WATERMARKS)*FULL_PERCENTAGE
	return codeTaken,extractionRate

def recursiveEmbed(code, mode, flag, inputPin):
	filename = "config.txt.encrypted"
	allGridPositions = checkInputFlagState(flag, filename)

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
			print("Embed key : " + str(innerKey) + " in Block " + str(index + 1))
			innerSip = encodeInteger(innerKey)
			innerSips.append(innerSip)
			blockParams = [innerSip, len(innerSip), innerKey]
			grid_cell = channel_array[offsetY:(offsetY + blockHeight), offsetX:(offsetX + blockWidth)]
			g_cell = Image.fromarray(grid_cell)
			optimalCValue,watermarkedBlock,gridParams = findBestCValues(blockParams, blockWidth, blockHeight, em, code, g_cell, mode, 1, flag, index, allGridPositions)
			optimalCValues.append(optimalCValue)
			watermarkedBlocks.append(watermarkedBlock)
			index = index + 1
	watermarkedImage = mergeCellsToImage(watermarkedBlocks, blockWidth, blockHeight, int(size))
	watermarkedImage = Image.fromarray((cv2.resize(np.array(watermarkedImage), (M,N), interpolation = cv2.INTER_AREA)))
	wimgName = "watermarked_" + IMAGE_NAME
	subpath,mappingPath = saveWatermarkedImage(wimgName, watermarkedImage, mappingDictionary, code, codeSips, flag)
	if(checkForFirstRun(flag, filename)):
		writeGridPlacementsInFile(optimalGridPositionForEachBlock)
		allGridPositions = optimalGridPositionForEachBlock
	psnr,ssim = getPSNRAndSSIM(np.array(watermarkedImage), channel_array)
	return watermarkedBlocks,codeSips,mappingDictionary,innerSips,subpath,optimalCValues,allGridPositions,gridParams

def extractRecursiveSip(watermarkedBlock, imagePath, originalKey, innerSip, code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock):
	em,ex = init()
	img = openImage(imagePath)
	M,N = img.size
	channel_array = np.array(img)

	sip1,sip2,sip3 = ex.getSip(watermarkedBlock, len(innerSip), PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	print("Key which was embeded :", originalKey)
	key1 = decodeSip(sip1)
	key2 = decodeSip(sip2)
	key3 = decodeSip(sip3)
	if(key1 == originalKey or key2 == originalKey or key3 == originalKey) :
		return 1,key1,key2,key3
	return 0,key1,key2,key3

def findBestCValues(blockParams, blockWidth, blockHeight, embedObject, code, g_cell, mode, extractionIsPrioritized, flag, index, allGridPositions) :
	gridSize,RBWidth,Rxy,Bxy = calculateBasicValues(blockWidth, blockHeight, blockParams[1], PR, PB)
	gridParams = [gridSize, RBWidth, Rxy, Bxy]
	if(mode == FAST_MODE):
		if(flag == 1):
			randomGridPosition = calculateRandomGridPosition(blockParams, blockWidth, blockHeight, gridSize)
			optimalGridPositionForEachBlock.append(randomGridPosition) 
			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, IMAGE_NAME, IMAGE_PATH, randomGridPosition)
			return optimalCValue,watermarkedBlock,gridParams
		else:
			gridPosition = allGridPositions[index]
			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, IMAGE_NAME, IMAGE_PATH, gridPosition)
			return optimalCValue,watermarkedBlock,gridParams
	else:
		optimalCValue,watermarkedBlock,optimalGridPosition = optimizeCValueFull(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, IMAGE_NAME, IMAGE_PATH, blockWidth, blockHeight)
		optimalGridPositionForEachBlock.append(optimalGridPosition)
		return optimalCValue,watermarkedBlock,gridParams



def main(image_path, code, flag, input_pin):
    global IMAGE_PATH, IMAGE_NAME
    IMAGE_PATH = image_path
    IMAGE_NAME = (((IMAGE_PATH.split("/"))[-1]).split(".jpg"))[0]
    
    start = time.time()
    code = getListFromCode(code)
    input_pin = getListFromCode(input_pin)

    # Run Main Algorithm
    watermarkedBlocks, codeSips, codeMapping, innerSips, subpath, optimalCValues, allGridPositions, gridParams = recursiveEmbed(code, FAST_MODE, flag, input_pin)
    # Return results or print them if needed
    return {"watermarkedBlocks": watermarkedBlocks, "codeSips": codeSips,"codeMapping":codeMapping,"innerSips": innerSips ,"subpath": subpath, "optimalCValues": optimalCValues, "allGridPositions": allGridPositions, "gridParams":gridParams}

if __name__ == "__main__":
    image_path = sys.argv[1]
    code = sys.argv[2]
    flag = int(sys.argv[3])
    input_pin = sys.argv[4]
    main(image_path, code, flag, input_pin)