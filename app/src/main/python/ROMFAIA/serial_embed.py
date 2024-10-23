import os,sys
import numpy as np
import embed_from_app
import extract_from_app
import hash_TD
import random
import time



def serialized_embed(path,keys,k,im_name,c,gs,id,out_path,size,option,root_path,map):
	print("-- starting watermarking algorithm (1st stage of ROMFAIA) --\n")
	img = repetitive_embed(path,keys,k,im_name,c,gs,id,root_path)
	print("-- starting tamper detection algorithm (2nd stage of ROMFAIA) --\n")
	t_img = tamper_embed(img,size,option,im_name,out_path,root_path,map)

	return t_img

def repetitive_embed(path,keys,k,im_name,c,gs,id,root_path):
	img = embed_from_app.startEmbeding(path,keys,k,im_name,c,gs,id,root_path)
	return img

def tamper_embed(path,size,option,name,out_path,root_path,map):
	model = hash_TD.HashTD(map)
	im = model.embed_hashes(path,name,size,out_path,root_path)

	return im


def serialized_extract(path,keys,k,gs,map,root_path,name,size):
	ex,ex_sips,s_rate = extract_from_app.startExtracting(path,keys,gs,k)
	print("Extraction rate from watermarking algorithm: ",s_rate)
	model = hash_TD.HashTD(map)
	message = model.extract_hashes(path,name,size,map,root_path)
	print("Validation from tamper detection algorithm: ",message)

