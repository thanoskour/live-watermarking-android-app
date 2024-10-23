from embed_key import EmbedPermutation
from extract_sip import ExtractPermutation
from PIL import Image
from encodeinteger import encodeInteger

def init():
	w = EmbedPermutation()
	ex = ExtractPermutation()
	return w,ex

def openImage(path):
		img = Image.open(path)
		return img

def getGeneralSipSize(generalKey) :
	SIP = encodeInteger(generalKey)
	SIZE = len(SIP)
	return SIP,SIZE

def checkCellsOfImage(SIP, SIZE) :
	sip_cells = []
	A_matrix = []
	for i in range(0,len(SIP)):
		row = []
		for j in range(0,SIZE):
			if(j == SIP[i] - 1):
				row.append("*")
				sip_cells.append((i,j))
			else:
				row.append("-")
		A_matrix.append(row)
	return sip_cells,A_matrix