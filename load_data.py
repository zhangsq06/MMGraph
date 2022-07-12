import numpy as np
import csv
import sys
import os
from os import path
##########################
SIZE = 5
############shuffle randomly all bases within a positive sequence to generate a negative sequence##############
def dinucshuffle(sequence):
	b = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
	np.random.shuffle(b)
	d = ''.join([str(x) for x in b])
	return d
#############
def modifystr(str, size=0):
  n = len(str)
  return str[size:n-size]
############################load training data###############
def load_encode_train(tfid, limit=5000,):
	seq_suffix =".seq"
	filename = "test_data/data/encode_101/%s_AC%s" % (tfid, seq_suffix)
	sequences=[]
	targets=[]
	header=True
	count=0
	tmp=[]
	with open(filename) as f:
		for line in f.readlines():
			if header:
				header=False
				continue
			count+=1
			if count >= limit:
				break
			line = line.rstrip().split('\t')
			seq = line[2]
			seq = modifystr(seq)
			target = float(line[-1])
			nseq = dinucshuffle(seq)
			sequences.append([seq, 1])
			sequences.append([nseq, 0])		
	return sequences
###########################load testing data#############
def load_encode_test(tfid):
	seq_suffix =".seq"
	filename = "test_data/data/encode_101/%s_B%s" % (tfid, seq_suffix)
	sequences=[]
	targets=[]
	header=True
	with open(filename) as f:
		lines = f.readlines()
		for line in lines[:5000]:
			if header:
				header=False
				continue
			line = line.rstrip().split('\t')
			if len(line) < 4:
				continue
			seq = line[2]
			seq = modifystr(seq)
			nseq = dinucshuffle(seq)
			sequences.append([seq, 1])
			sequences.append([nseq, 0])
	return sequences

def load_motif_seq(tfid):
	seq_suffix =".seq"
	filename = "test_data/data/encode_101/%s_B%s" % (tfid, seq_suffix)
	sequences=[]
	targets=[]
	header=True
	with open(filename) as f:
		lines = f.readlines()
		for line in lines[:2000]:
			if header:
				header=False
				continue
			line = line.rstrip().split('\t')
			if len(line) < 4:
				continue
			seq = line[2]
			seq = modifystr(seq)
			sequences.append(seq)
	return sequences

if __name__=='__main__':
	sequences=load_encode_train('BRF2_K562_BRF2_Harvard')
	print(sequences)
