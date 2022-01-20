# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:05:21 2019

@author: Shuangquan Zhang, Cankun Wang
"""

import os
import gzip
import argparse
import csv
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',default='',help='fa')
args = parser.parse_args()

def getchrom_index(name):
    path='./inputs/TfbsUniform_hg19_ENCODE/'
    name1 = name.split('.')[0]

    with gzip.open(path+name+'.narrowPeak.gz', 'rt') as fnarrow_peak:
        peak_info = list(csv.reader(fnarrow_peak, delimiter = '\t'))
    peak_num = len(peak_info)
    print(peak_num)    # Number of peaks
    test_val=['train.bed', 'test.bed']
    test_val_fa=[name1+'_AC.fa', name1+'_B.fa']

    file1=open(test_val[0],'w')
    file2=open(test_val[1],'w')

    for j in range(peak_num):
        t1=peak_info[j]
#        print(t)
#        t1=t.split('\t')
        t11=round((int(t1[1])+int(t1[2]))/2)-49
        t12=round((int(t1[1])+int(t1[2]))/2)+52
        tt=str(t1[0])+'\t'+ str(int(t11)) + '\t' + str(int(t12)) +'\n'
        if j % 2 ==0:
            file1.writelines(tt)
        elif j<= 1000:
            file2.writelines(tt)
    file1.close()
    file2.close()

    cmd='bedtools getfasta -fi ./GRCh37.p13.genome.fa -bed '+ test_val[0]+ ' -s -fo ./data/'+test_val_fa[0]
    os.system(cmd)
    cmd='bedtools getfasta -fi ./GRCh37.p13.genome.fa -bed '+ test_val[1]+ ' -s -fo ./data/'+test_val_fa[1]
    os.system(cmd)
    os.remove(test_val[0])
    os.remove(test_val[1])

if __name__=='__main__':
    path='./TfbsUniform_hg19_ENCODE/'
    files=os.listdir(path)
    for name in files[:1]:
        getchrom_index(name)