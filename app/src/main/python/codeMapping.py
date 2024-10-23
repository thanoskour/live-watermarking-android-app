import random
import math

LENGTH = 36

def getBlockDimensions(M, N, SIZE) :
	blockWidth = math.floor((M / SIZE))
	blockHeight = math.floor((N / SIZE))
	return blockWidth,blockHeight

def getListFromCode(code):
	listCode = []
	for i in range(len(code)):
		if(code[i] not in ['a','b','c','d','e','f','-']):
			listCode.append(int(code[i]))
		elif(code[i] == '-'):
			continue
		else:
			listCode.append(code[i])
	return listCode

def getCodeMapping(code):
	listOfNums = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']
	listOfSips = list(range(16, 32))
	dictionary = {}
	for i in range(len(listOfSips)):
		dictionary[listOfSips[i]] = listOfNums[i]
	return dictionary

def getSipsFromCode(dictionary, code):
	codeSips = []
	keyList = list(dictionary.keys())
	valuesList = list(dictionary.values())
	for i in range(len(code)):
		position = valuesList.index(code[i])
		sip = keyList[position]
		codeSips.append(sip)
	return codeSips

def checkForFullCode(code):
	diff = LENGTH - len(code)
	if(len(code) < LENGTH):
		for i in range(diff):
			code.append(random.randint(0,9))
	else:
		code = code[0:LENGTH]
	return code
