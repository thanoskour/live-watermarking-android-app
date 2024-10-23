import cv2,sys,re,time,random,numpy as np,matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from decodesip import decodeSip
from recossip import recsip
from math import log10, sqrt
from PIL import Image
#from metrics import *
import metrics
from initializer import *
from codeMapping import *
import rsipw
#from rsipw import extractRecursiveSip

PR = 2
PB = 2

def optimizeCValueFast(
						blockParams, 
						embedObject, 
						code, 
						g_cell, 
						extractionIsPrioritized, 
						gridSize, 
						RBWidth, 
						Rxy, 
						Bxy, 
						imageName, 
						imagePath, 
						gridPositionForEachBlock
					):
	low,high = 0,100
	originalHigh,optimalCValue,isExtractedPrevious,stop,psnr,ssim,isExtracted,completeExtractions = high,high,0,0,0,0,0,[]
	print("Running with c = 1")
	watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, 1, PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	psnr,ssim = getPSNRAndSSIM(g_cell, watermarkedBlock)
	isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams[2], blockParams[0], code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	if(isExtracted == 1):
		return 1,watermarkedBlock,isExtracted,psnr,ssim
	while(low <= high):
		print("Running with c = " + str(optimalCValue))
		watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, optimalCValue, PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
		psnr,ssim = getPSNRAndSSIM(g_cell, watermarkedBlock)
		isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams[2], blockParams[0], code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
		if(stop == 1):
			break
		if(isExtractedPrevious == 1 and isExtracted == 0 and optimalCValue < 7):
			optimalCValue = optimalCValuePrevious
			stop = 1
			continue
		isExtractedPrevious = isExtracted 
		optimalCValuePrevious = optimalCValue
		if(optimalCValue == 0) :
			break
		if(isExtracted == 1):
			step = (high - low) // 2
			optimalCValue = high - step
			high = step
			completeExtractions.append(optimalCValuePrevious)
		elif(isExtracted == 0 and optimalCValue >= 25):
			step = (high - low) // 2
			optimalCValue = high - step
			high = step
		elif(isExtracted == 0 and optimalCValue < 25) :
			step = (high - low) // 2
			optimalCValue = high + step
			low = high
			high = optimalCValue
		if(optimalCValuePrevious == optimalCValue) :
			break
	if((isExtracted == 0) and (extractionIsPrioritized == 1)):
		if(completeExtractions != []):
			optimalCValue = optimalCValue + 1
			globalCValue = min(completeExtractions)
			low = optimalCValue
			high = globalCValue
			while(low <= high):
				print("Running with c = " + str(optimalCValue))
				if(optimalCValue > globalCValue):
					optimalCValue = globalCValue
					watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, optimalCValue, PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
					break
				watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, optimalCValue, PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams[2], blockParams[0], code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				optimalCValuePrevious = optimalCValue
				if(isExtracted == 1):
					break
				elif(isExtracted == 0):
					low = optimalCValue
					optimalCValue = optimalCValue + 3
				if(optimalCValuePrevious == optimalCValue) :
					break
		else:
			low = optimalCValue + 1
			startCValue = low
			high = 100
			optimalCValue = startCValue
			while(low <= high):
				print("Running with c = " + str(optimalCValue))
				watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, optimalCValue, PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				isExtracted = getExtractionResult(watermarkedBlock, imagePath, blockParams[2], blockParams[0], code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
				optimalCValuePrevious = optimalCValue
				if(isExtracted == 1):
					break
				elif(isExtracted == 0):
					low = optimalCValue
					optimalCValue = optimalCValue + 3
				if(optimalCValuePrevious == optimalCValue) :
					break
			if(isExtracted == 0):
				print("Running with c = 1")
				optimalCValue = 1
				watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, optimalCValue, PR, PB, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	printResults(2, optimalCValue, 0, 0, [])
	return optimalCValue,watermarkedBlock,isExtracted,psnr,ssim

def optimizeCValueFull(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imageName, imagePath, blockWidth, blockHeight):
	optimalCValuesForGrid,optimalGridPositions,optimalPSNR,optimalSSIM,extractionResults = [],[],[],[],[]
	for offsetY in range(0, (blockHeight - (gridSize[1] * blockParams[1]) + (gridSize[1])), (gridSize[1])):
		if(offsetY >= ((blockHeight) - (gridSize[0] * blockParams[1]))):
			break
		for offsetX in range(0, (blockWidth - (gridSize[0] * blockParams[1]) + (gridSize[0])), (gridSize[0])):
			gridPositionForEachBlock = [offsetY,offsetX]
			optimalCValue,watermarkedBlock,isExtracted,psnr,ssim = optimizeCValueFast(blockParams, embedObject, code, g_cell, extractionIsPrioritized, gridSize, RBWidth, Rxy, Bxy, imageName, imagePath, gridPositionForEachBlock)
			optimalCValuesForGrid.append(optimalCValue)
			extractionResults.append(isExtracted)
			optimalPSNR.append(psnr)
			optimalSSIM.append(ssim)
			optimalGridPositions.append(gridPositionForEachBlock)
			printResults(2, optimalCValue, 0, 0, [])
	currentC,currentPSNR,currentSSIM,bestIndex = 30,0,0,0
	for i in range(len(extractionResults)):
		if(extractionResults[i] == 1):
			if(optimalCValuesForGrid[i] <= currentC):
				currentC = optimalCValuesForGrid[i]
				if(optimalPSNR[i] >= currentPSNR and optimalSSIM[i] >= currentSSIM):
					currentPSNR = optimalPSNR[i]
					currentSSIM = optimalSSIM[i]
					bestIndex = i
	optimalGridPosition = optimalGridPositions[bestIndex]
	printResults(1, currentC, currentPSNR, s, optimalGridPosition)
	watermarkedBlock = embedObject.getWatermarkedImage(g_cell, blockParams[0], blockParams[1], imageName, currentC, 2, 2, gridSize, RBWidth, Rxy, Bxy, optimalGridPosition) 
	return currentC,watermarkedBlock,optimalGridPosition

def calculateRandomGridPosition(blockParams, blockWidth, blockHeight, gridSize):
	rowSum,columnSum,moveRow,moveColumn = 0,0,[0],[0]
	rowLimit = math.floor(((blockHeight) - (gridSize[0] * blockParams[1])) / (gridSize[0]))
	columnLimit = math.floor(((blockWidth) - (gridSize[1] * blockParams[1])) / (gridSize[1]))
	for i in range(rowLimit):
		rowSum = rowSum + gridSize[0]
		moveRow.append(rowSum)
	for j in range(columnLimit):
		columnSum = columnSum + gridSize[1]
		moveColumn.append(columnSum)
	rowIndex = random.randint(0, (len(moveRow)) - 1)
	columnIndex = random.randint(0, (len(moveColumn)) - 1)
	offsetY = moveRow[rowIndex]
	offsetX = moveColumn[columnIndex]
	gridPositionForEachBlock = [offsetY,offsetX]
	return gridPositionForEachBlock

def getPSNRAndSSIM(g_cell, watermarkedBlock):
	psnr = metrics.PSNR(np.array(g_cell), np.array(watermarkedBlock))
	ssim = metrics.SSIM(np.array(g_cell), np.array(watermarkedBlock))
	print("PSNR taken " + "{:.5f}".format(psnr))
	print("SSIM taken " + "{:.5f}".format(ssim))
	return psnr,ssim

def getExtractionResult(watermarkedBlock, compressedImageName, innerKey, sip, code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock):
	isExtracted,key1,key2,key3 = rsipw.extractRecursiveSip(watermarkedBlock, compressedImageName, innerKey, sip, code, gridSize, RBWidth, Rxy, Bxy, gridPositionForEachBlock)
	print("Result of extraction =",isExtracted)
	print("")
	return isExtracted

def printResults(mode, c, psnr, ssim, bm):
	if(mode == 1):
		print("Running with c = " + str(c))
		print("PSNR = " + str(psnr))
		print("SSIM = " + str(ssim))
		print(bm)
		print("Finished grid movement")
		print("")
	elif(mode == 2):
		print("Optimal c value for grid = " + str(c))
		print("-----------------------------------")
		print("")
	elif(mode == 3):
		if(psnr != 0):
			print("Complete Extraction in block " + str(c + 1) + "\n")
		else:
			print("Could not extract in block " + str(c + 1) + "\n")
	else:
		print("Extraction percentage = " + str(c) + "%\n")