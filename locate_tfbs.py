# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:09:50 2022

@author: ShuangquanZhang
"""
import scipy
from utils import *
from load_data import load_encode_test, load_encode_train,load_motif_seq
import numpy as np
import os
from com_mi import calc_MI, cal_IC
import sys
import argparse
import math
################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--dataset', default='GSE114204_encode',type=str,help='The prefix name of the dataset')
parser.add_argument('--k', default=5,type=int,help='the length of k-mer')
args = parser.parse_args()
# ##############################
def construct_vocab(seqs, size, thresh = 0):
    word_vocab = {}
    word_freq = {}
    for i, seq in enumerate(seqs):
        subseqs = [seq[i:i+size] for i in range(0, len(seq) - size + 1, 1)]
        word_vocab[seq]=subseqs
        for subseq in subseqs:
            if subseq not in list(word_freq.keys()):
                word_freq[subseq] = 1
            else:
                word_freq[subseq]+=1
    return word_vocab,word_freq
### 
def tf_ks(seq,size=5):
    sword_freq = {}
    for i in range(0,len(seq)-size+1,1):
        subseq = seq[i:i+size]
        if subseq not in sword_freq:
            sword_freq[subseq] = 1
        else:
            sword_freq[subseq]+=1
    return sword_freq
##################nomralization###################
def standardization(data):
	data = np.array(data)
	mu = np.mean(data, axis=0)
	sigma = np.std(data, axis=0)
	nor_data = (data-mu)/sigma 
	nor_data = list(nor_data)
	return nor_data
###################co-occurrence edge######################
def Acoo_value(nums,K12,K1,K2):    
    if K12>0:
        pk1k2 = K12/nums
        pk1 = (K1+K12)/nums
        pk2 = (K2+K12)/nums
        Acoo_k1k2 = np.log(pk1k2/(pk1*pk2+1e-3))
    else:
        Acoo_k1k2 = 0
    weight = Acoo_k1k2
    return weight
#####calculate the Wco matrix    
def Acoo(seqs,word_freq,word_vocab,threshholds=0):
    keys=list(word_freq.keys())#KMERS
    nums = len(seqs)
    rows_cols = [[i,j] for i in range(len(keys)) for j in range(i, len(keys))]
    kmers = [[keys[i],keys[j]] for i in range(len(keys)) for j in range(i, len(keys))]
    rows=np.random.rand(len(kmers))
    cols=np.random.rand(len(kmers))
    weights=np.random.rand(len(kmers))

    for t in range(len(kmers)):
        K12 = len(['s' for seq  in seqs if kmers[t][0] in word_vocab[seq] and kmers[t][1] in word_vocab[seq]])
        if K12 > threshholds:
            K1 = len(['s' for seq  in seqs if kmers[t][0] in word_vocab[seq] and kmers[t][1] not in word_vocab[seq]])
            K2 = len(['s' for seq  in seqs if kmers[t][0] not in word_vocab[seq] and kmers[t][1] in word_vocab[seq]])
            weight = Acoo_value(nums, K12, K1, K2)
            weights[t] = weight
            rows[t]= rows_cols[t][0]
            cols[t] = rows_cols[t][1]
    node_size = len(keys)
    weights = standardization(weights)
    adj = scipy.sparse.csr_matrix((weights, (list(rows), list(cols))),shape=(node_size, node_size))
    # adj = adj.T+adj
    Acoo = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return Acoo
################load k-mer and sequence#############################
def generateAdjs(tfids,Kmers=5):
    test_seqs = load_encode_test(tfids)
    train_seqs = load_encode_train(tfids)
    seqs = train_seqs + test_seqs
    seqs = [seq[0] for seq in seqs]
    pos_seqs = [t[0] for t in test_seqs]
    
    motif_seqs = load_motif_seq(tfids)
    neg_seqs = [t for t in pos_seqs if t not in motif_seqs ] 

    word_vocab,word_freq = construct_vocab(seqs, size=Kmers)
    strsize=str(Kmers)
    ext = ".npz"
    filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_coo" + ext
    if os.path.exists(filename):
        filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_coo" + ext
        adj_coo = scipy.sparse.load_npz(filename)
    else:
        adj_coo = Acoo(seqs, word_freq, word_vocab, threshholds=0)
    return adj_coo.toarray(), word_vocab, word_freq,motif_seqs,neg_seqs
####load the embedding of k-mer and sequences
def load_embed(name,Kmers):
    seq_path = './data/TFBS/'+name+str(Kmers)+'_seq.npy'
    kmer_path = './data/TFBS/'+name+str(Kmers)+'_kmer.npy'
    seq_embed = np.load(seq_path)
    kmer_embed = np.load(kmer_path)
    return seq_embed,kmer_embed
def construct_data(data,values):
    datas={}
    for i in range(len(data)):
        datas[data[i]]=values[i,:]
    return datas
################################################
def TFBSs(tfid, Kmers):
    seq_embed,kmer_embed = load_embed(tfid, Kmers)
    adj_coo, word_vocab, word_freq, motif_seqs,neg_seqs = generateAdjs(tfid,Kmers)
    all_kmers = list(word_freq.keys())
    #计calculate the MI between sequence node and k-mer node.
    kmer_s=list(word_freq.keys())
    seqs=list(word_vocab.keys())
    seqs_data=construct_data(seqs,seq_embed)
    kmers_data=construct_data(kmer_s,kmer_embed)
    neg_nmis=[]
    for j in range(len(neg_seqs)):
        neg_seq_embeding = seqs_data[neg_seqs[j]]
        neg_seq_mers = word_vocab[neg_seqs[j]]
        for k in range(len(neg_seq_mers)):
            kmer_embeding = kmers_data[neg_seq_mers[k]]
            neg_nmi = calc_MI(neg_seq_embeding, kmer_embeding)
            neg_nmis.append(neg_nmi)
    neg_average_nmi = np.mean(np.array(neg_nmis)) #mean(MI0)
####################################
    ranges=[neg_average_nmi]
    for vals in ranges:
        # print('vals is %5f' % vals)
        seqname = './motifs/'+tfid+'_'+str(round(vals,2))+'_.txt'
        faname = './motifs/'+tfid+'_'+str(round(vals,2))+'_.fa'
        file1=open(seqname,'w+')
        file2=open(faname,'w+')
        count =0
        for i in range(len(motif_seqs[:])):
            seq_embeding = seqs_data[motif_seqs[i]] ## sequences embedding
            seq_mers = word_vocab[motif_seqs[i]]
            for j in range(len(seq_mers)):
                kmer_embeding = kmers_data[seq_mers[j]] ## k-mer embedding
                nmi = calc_MI(seq_embeding, kmer_embeding) ##MI1(p,i)
                if nmi > vals: ###if MI1(p,i)>mean(MI0)
                    cp_index = int(j+math.ceil((Kmers-1)/2)) #startkp
                    right_frag = int(cp_index+Kmers)
                    left_frag = int(cp_index-Kmers+1)                   
                    if right_frag < 101 and left_frag>0:
                        kr_index = all_kmers.index(motif_seqs[i][cp_index+1:right_frag]) #kr(p,i)
                        kl_index = all_kmers.index(motif_seqs[i][left_frag:cp_index]) #kl(p,i)
                        coexisting_proba = adj_coo[kl_index, kr_index] ##coexisting probability of kl(p,i) and kr(p,i)
                        if coexisting_proba > 0.5:
                            seq = motif_seqs[i][left_frag:right_frag]
                            file1.writelines(seq+'\n')
                            strs='>'+'seq'+'_'+str(i)+'_'+str(left_frag)+'_'+str(right_frag)+'\n'
                            file2.writelines(strs)
                            file2.writelines(seq+'\n')
                            count += 1
        file1.close()
        file2.close()
        # information_content = cal_IC(faname)
        # if (sum((np.array(information_content) >= 1) * 1) < 1) or count < 5:
        #     print('remove '+faname)
        #     os.system("rm -rf " + faname)
        #     os.system("rm -rf " + seqname)
if __name__=='__main__':
    TFBSs(args.dataset, args.k)
