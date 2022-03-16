# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:15:53 2022

@author: ShuangquanZhang
"""

import argparse
import os
import re
###########################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--bed', default='GSE114204',type=str,help='The prefix name of the bed file')
parser.add_argument('--path', default='',type=str,help='Path to the bed file') #contain hg38.fa GSE.bed GSE.bam  path/
parser.add_argument('--peak_flank',type=int, default=50, help='peaks length')
args = parser.parse_args()
######################################################
def getchrom_index(args):
    path = args.path
    name = args.bed  #GSE114204
    hint_tobias_name = path + "final_hint_tobias/" + name + "_final_hint_tobias.bed"
    file = open(hint_tobias_name, 'r')
    data1 = file.readlines()
    file.close()
    ll = len(data1)
    test_val = ['train.bed', 'test.bed']
    test_val_fa = [name + 'train1.fa', name + 'test1.fa']
    pattern = re.compile(r'chr')
    file0 = open(test_val[0], 'w')
    file1 = open(test_val[1], 'w')
    for j in range(ll):
        t = data1[j]
        match_res = pattern.match(t)
        t1 = t.split('\t')
        if match_res:
            t11 = round((int(t1[1]) + int(t1[2])) / 2) - args.peak_flank + 1
            t12 = round((int(t1[1]) + int(t1[2])) / 2) + args.peak_flank + 1
            if t11 > 0 and t12 > 0:
                tt = str(t1[0]) + '\t' + str(int(t11)) + '\t' + str(int(t12)) + '\n'  # + '\t' + str(t1[4]) + '\n'
                if j <= 1000:
                    if j % 2 == 0:
                        file0.writelines(tt)
                    else:
                        file1.writelines(tt)
                else:
                    file0.writelines(tt)
    file0.close()
    file1.close()
#################################
    for k in range(2):
        cmd = 'bedtools getfasta -fi '+path+'hg38.fa -bed ' + test_val[k] + ' -fo ' + test_val_fa[k]
        os.system(cmd)
        os.remove(test_val[k])