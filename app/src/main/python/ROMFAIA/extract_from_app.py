#from .decodesip import decodeSip
#from .embed_key import EmbedPermutation
#from .extract_sip import ExtractPermutation
#from .encodeinteger import encodeInteger
#from .recossip import recsip
import decodesip
import embed_key
import extract_sip
import encodeinteger
import recossip
import sys
import re
import rsipw
#from .rsipw import extract_recsip
import ast
import hashlib

def validate_hash(h1,h2):
	if(h1 == h2):
		return "Hash maches with initial random key"
	else:
		return "Hash does not mach Tampering Detected"

def read_hashes(ex_keys):
	h1 = (open("private_hash.txt","r")).readline()
	h2 = hashlib.new('sha256')
	h2.update(bytes(''.join(ex_keys),'ascii'))
	h2 = h2.hexdigest()

	return h1,h2


def startExtracting(im,keys,gs,k):

	img_format = im.split(".")[1]
	print("Image format: ",img_format)
	ex,ex_sips,s_rate,code = rsipw.extract_recsip(im,keys,2,2,gs,k,img_format)
	print("Extracted code: ",code)
	print("Success rate: ",s_rate,"%")




	return ex,ex_sips,s_rate



