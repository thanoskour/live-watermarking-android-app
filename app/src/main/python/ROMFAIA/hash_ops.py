import os,sys
import numpy as np
import hashlib

look_up_hex = { 'a':10,
                'b':11,
                'c':12,
                'd':13,
                'e':14,
                'f':15 }
look_up_hex_inv = { 10:'a',
                    11:'b',
                    12:'c',
                    13:'d',
                    14:'e',
                    15:'f'}


def hash_to_bits(hash_val):
	ch = [i for i in hash_val]
	bits = []
	for j in range(len(ch)):
		if(ch[j].isalpha()):
			bi = ''.join(format(look_up_hex[ch[j]], '04b'))
		else:
			bi = ''.join(format(int(ch[j]), '04b'))
		for b in bi:
			bits.append(b)
	return bits

def bits_to_hash(bit_seq):
	hash_val_chr = []
	for i in range(0,len(bit_seq),4):
		word = bit_seq[i:i+4]
		code_word = ''.join(word)
		dec_val = int(code_word,2)
		if(dec_val >= 10):
			hash_val_chr.append(look_up_hex_inv[dec_val])
		else:
			hash_val_chr.append(str(dec_val))

	hash_val = ''.join(hash_val_chr)
	return hash_val

def xor_hashes(hashes):
	hash_bits = []
	for i in range(len(hashes)):
		bits = []
		ch = [j for j in hashes[i]]
		for k in range(len(ch)):
			if(ch[k].isalpha()):
				bi = ''.join(format(look_up_hex[ch[k]], '04b'))
			else:
				bi = ''.join(format(int(ch[k]), '04b'))
			for b in bi:
				bits.append(b)
			#print(len(bits))
		hash_bits.append(bits)

	hash_bits_arr = np.asarray(hash_bits)
	hash_bits_arr_T = hash_bits_arr.T

	return hash_bits_arr_T


def combine_hashes_and_hash(hash1,hash2):
	con_hash = hash1 + hash2
	s_bytes = bytes(con_hash, 'ascii')
	h = hashlib.new('sha256')
	h.update(s_bytes)
	hashb = h.hexdigest()

	return hashb
