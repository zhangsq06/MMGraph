# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:56:29 2022

@author: ShuangquanZhang
"""
import argparse

import os
####################################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--bed', default='GSE114204',type=str,help='The prefix name of the bed file')
parser.add_argument('--path', default='',type=str,help='Path to the bed file') #contain hg38.fa GSE.bed GSE.bam  path/
parser.add_argument('--peak_flank',type=int, default=50, help='peaks length')
args = parser.parse_args()
###########################################
#################################
def makeseq(input_name, out_name):
    file = open(input_name, 'r')
    data1 = open(out_name, 'w')
    data = file.readlines()
    file.close()
    times = 0
    for t1 in data:
        if t1.startswith('>'):
            tt = 0
        else:
            ttt = 'A' + '\t' + 'peaks' + '\t' + t1.strip().upper() + '\t' + '1' + '\n'
            data1.writelines(ttt)
            times += 1
    data1.close()

###########################################
def get_data(args):
    path = args.path
    name = args.bed  # GSE114204
    innames = [name + 'train1.fa', name + 'test1.fa']
    outnames = [name + '_encode_AC.seq', name + '_encode_B.seq', name + '_encode.seq']
    for i in range(2):
        makeseq(innames[i], outnames[i])
        cmd = 'cat ' + outnames[i] + ' >> ' + outnames[2]
        os.system(cmd)
        cmd = 'gzip ' + outnames[i]
        os.system(cmd)
        path_encode = path + 'data/encode_' + str(args.peak_flank * 2 + 1)
        if not os.path.exists(path_encode):
            os.mkdir(path_encode)
        cmd = 'mv ' + outnames[i] + '.gz ' + path_encode
        os.system(cmd)
    cmd = 'gzip ' + outnames[2]
    os.system(cmd)
    path_encode = path + 'data/encode_'+ str(args.peak_flank * 2 + 1)
    if not os.path.exists(path_encode):
        os.mkdir(path_encode)
###############################################
    cmd = 'mv ' + outnames[2] + '.gz ' + path_encode
    os.system(cmd)
    path_txt = path + 'data/encode_tfbs.txt'
    file = open(path_txt, 'a+')
    txxt = name + '_encode' + '\t' + name + '_encode' + '\n'
    file.writelines(txxt)
    file.close()
    os.remove(innames[0])
    os.remove(innames[1])