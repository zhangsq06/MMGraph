# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:13:16 2022

@author: ShuangquanZhang
"""
import argparse

import os
import re
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--bed', default='GSE114204',type=str,help='The prefix name of the bed file')
parser.add_argument('--path', default='',type=str,help='Path to the bed file') #contain hg38.fa GSE.bed GSE.bam  path/
args = parser.parse_args()

def get_peaks(args):
    path = args.path # path
    name = args.bed # GSE114204
    hint_tobias_name=path+"final_hint_tobias/"+name+"_final_hint_tobias.bed"
    file = open(hint_tobias_name, 'r')
    data = file.readlines()
    file.close()
    ll = len(data)
    # name = '../data/TfbsUniform_hg19_ENCODE/' + name + '_encode.narrowPeak' #../data/TfbsUniform_hg19_ENCODE/GSE114204_encode.narrowPeak
    path = path + 'data'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/TfbsUniform_hg38_ENCODE'
    if not os.path.exists(path):
        os.mkdir(path)
    name = path+'/' + name + '_encode.narrowPeak'
    file1 = open(name, 'w')
    pattern = re.compile(r'chr\d+')
    for i in range(ll):
        t = data[i]
        match_res = pattern.match(t)
        if match_res:
            t1 = t.split('\t')
            tt = str(t1[0]) + '\t' + str(t1[1]) + '\t' + str(t1[2]) #+ '\t' + str(t1[4])
            file1.writelines(tt)
    file1.close()
    cmd = 'gzip ' + name
    os.system(cmd)
if __name__ == '__main__':
    get_peaks(args)