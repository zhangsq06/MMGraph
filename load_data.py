import numpy as np
import csv
import scipy.sparse
import sys
import os
from os import path
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

#########################################

SIZE = 10

def dinucshuffle(sequence):
	b = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
	np.random.shuffle(b)
	d = ''.join([str(x) for x in b])
	return d


def modifystr(str, size=0):
  n = len(str)
  return str[size:n-size]
def reverse(seq):
	dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'}
	rseq = [dictReverse[seq[c]] for c in range(len(seq))]
	rseq = ''.join(rseq)
	#print(seq, rseq)
	return rseq
############################

def load_encode_train(tfid, limit=50000, use_reverse=False):
	seq_suffix =".seq"
	filename = "./data/encode/%s_AC%s" % (tfid, seq_suffix)
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
			seq = modifystr(seq, SIZE)
			target = float(line[-1])
			if use_reverse:
				seq = reverse(seq)

			nseq = dinucshuffle(seq)
			sequences.append([seq, 1])
			sequences.append([nseq, 0])		
	return sequences

def load_encode_test(tfid, use_reverse=False):
	seq_suffix =".seq"
	filename = "./data/encode/%s_B%s" % (tfid, seq_suffix)
	sequences=[]
	targets=[]
	header=True
	with open(filename) as f:
		lines = f.readlines()
		for line in lines[:500]:
			if header:
				header=False
				continue
			line = line.rstrip().split('\t')
			if len(line) < 4:
				continue
			seq = line[2]
			seq = modifystr(seq, SIZE)
			if use_reverse:
				seq = reverse(seq)
				
			nseq = dinucshuffle(seq)
			sequences.append([seq, 1])
			sequences.append([nseq, 0])

	return sequences