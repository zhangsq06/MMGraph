# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:46:27 2021

@author: ShuangquanZhang
"""
# install scipy, scikit-learn, Biopython
from scipy.stats import chi2_contingency
import numpy as np
from Bio.SubsMat import FreqTable
from Bio import AlignIO
from Bio.Align import AlignInfo
from Bio.Alphabet import IUPAC
from sklearn.metrics import mutual_info_score
#calculate the mutual informtion
def calc_MI(x, y, bins=10):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    # mi = 0.5 * g / c_xy.sum()
    return mi

#calculate the information content
def cal_IC(path_motifInstance):
    
    EXPECT_FREQ = {"A": 0.25, 
                "G": 0.25, 
                "T": 0.25, 
                "C": 0.25}    
    e_freq_table = FreqTable.FreqTable(EXPECT_FREQ, FreqTable.FREQ, IUPAC.unambiguous_dna)
    information_content = []
    alignment = AlignIO.read(path_motifInstance, "fasta")
    summary_align = AlignInfo.SummaryInfo(alignment)
    for j in range(10):                
        information_content.append(summary_align.information_content(j, j + 1, e_freq_table = e_freq_table, chars_to_ignore = ['N']))
    return information_content
####################