from __future__ import division
from constants import *
import os
import gzip
import csv

import numpy as np


import collections
from Bio import SeqIO
from Bio import SeqUtils
from processing_peaks import getchrom_index

#PATH_ENCODE_TFBS='./encode_tfbs.txt'
data_names='./inputs/encode_tfbs.txt'
# The base-num pairs
BASE_NUM = {'A' : 1, 
            'C' : 2,
            'G' : 3,
            'T' : 4,
            'N' : 0
           }

PATH_DATA = ''
# Get human genome sequence and shape
path_genome = './GRCh37.p13.genome.fa'                 # Fasta files of hg38
genome_seq = {}                                      # Human genome sequence, where key = chromosome id, value = sequence
genome_shape = collections.defaultdict(dict)         # Human genome shape, where key = chromosome id, value = DNA shape
chrom_seqSize = {}                                   # Size of each chromosome, where key = chromosome id, value = length

# Read genome sequence from human genome
fasta_sequences = SeqIO.parse(open(path_genome),'fasta')
for fasta in fasta_sequences:
    if fasta.id.startswith('chr'):
        chrom_seqSize[fasta.id] = len(fasta)
        genome_seq[fasta.id] = fasta.seq

# Get ChIP-seq data list
def get_data_name(data_names):
    train_data_list = []
    table_name_list = []
    with open(data_names) as f:
        results = list(csv.reader(f, delimiter = '\t'))
        
    for i in range(len(results)):
        train_data_list.append(results[i][0].strip())
        table_name_list.append(results[i][1].strip())
    return train_data_list, table_name_list


# Coordinates of each peak
def get_peak_coor(tfbs_unif_path):
    # Get all chromosome included in this file 
    with gzip.open(tfbs_unif_path, 'rt') as fnarrow_peak:
        peak_info = list(csv.reader(fnarrow_peak, delimiter = '\t'))
    peak_num = len(peak_info)    # Number of peaks
    chr_set = set([peak_info[i][0] for i in range(len(peak_info))])
            
    # Construct a nested dictionalry to record gene position
    chrome = collections.defaultdict(dict)
    for chr_index in chr_set:
        chrome[chr_index]['start'] = []
        chrome[chr_index]['end'] = []

    # Record coordinates of each peak  
    for line in peak_info:
        chr_start = line[1]         # Start position of current peak
        chr_end = line[2]           # End position of current peak
    
        chrome[line[0]]['start'].append(int(chr_start))
        chrome[line[0]]['end'].append(int(chr_end))

    for chr_index in chr_set:
        chrome[chr_index]['start'] = np.array(chrome[chr_index]['start'])
        chrome[chr_index]['end'] = np.array(chrome[chr_index]['end'])
        
    return chrome, peak_num

# Load sequences, targets, and  from ENCODE
def load_data_encode(data_path, peak_coor, data_name, table_name,path_curr_data='', train_test="train", peak_flank=50, back_grou=['gc_match']):

    # Make dir for training or test data
#    make_dir(path_curr_data + "/" + train_test)
    peak_length = 2 * peak_flank + 1

        # Read the sequence file
    with gzip.open(data_path, 'rt') as f:
        sequenceList = list(csv.reader(f, delimiter = '\t')) 
        invalid_seq_index = [i for i, seq in enumerate(sequenceList[1:]) if 'N' in seq[2].strip()]  

    # If the number of valid sequences less than 10000, we generate (10000 - #valid sequences) sequences by resampling from the available sequences
    if train_test == "train" and len(sequenceList) - 1 - len(invalid_seq_index) < MINIMUM_POSI_SEQ:
        sample_index = np.random.choice(len(sequenceList) - 1 - len(invalid_seq_index), MINIMUM_POSI_SEQ - (len(sequenceList) - 1 - len(invalid_seq_index)), replace = True)
        sequenceList_valid = [sequenceList[1:][i] for i in range(len(sequenceList[1:])) if i not in invalid_seq_index]
        sample_sequence = [sequenceList_valid[sample_index[i]] for i in range(len(sample_index))]
        sequenceList = sequenceList + sample_sequence
        
    # Sequences (numpy array)
    sequences = np.array([[float(BASE_NUM[x]) for x in sequenceList[i][2]] for i in range(1, len(sequenceList))])

    # Sequences ('A', 'T', 'C', 'G')
    sequences_alph = [sequenceList[i][2] for i in range(1, len(sequenceList))]

    # Remove invalid sequences according to invalid_seq_index
    sequences_alph = np.array(np.delete(sequences_alph, invalid_seq_index, 0))

    if '_B.' in data_path:
        data_name='./data/'+table_name+'_B_neg'
    elif '_AC.' in data_path:
        data_name='./data/'+table_name+'_AC_neg'
    print(data_name)
    files=open(data_name+'.bed','w')
    # Generate negative sequences
    if train_test != 'all':
        seqs = []
        while len(seqs) < sequences.shape[0]:
            GC_content = SeqUtils.GC(sequences_alph[len(seqs)])                        # GC content of current DNA sequence 
            chr_index = list(chrom_seqSize.keys())[np.random.randint(len(chrom_seqSize))]
		#print 'length',chrom_seqSize[chr_index] - peak_length		
# Randomly select a chromosome
            chr_start = np.random.randint(chrom_seqSize[chr_index] + peak_length)      # Randomly select a start position in the selected chromosome
            chr_end = chr_start + peak_length                                          # The end position of the selected segment
            seq_segment = str(genome_seq[chr_index][chr_start : chr_end]).upper()

            # The sequences without overlapping and only consisting of 'ACGT' will be used as background sequence                    
            if ((chr_index not in peak_coor.keys()) or (not (np.any((peak_coor[chr_index]['start'] <= chr_start) & (chr_start <= peak_coor[chr_index]['end'])) or np.any((peak_coor[chr_index]['start'] <= chr_end) & (chr_end <= peak_coor[chr_index]['end'])) or np.any((peak_coor[chr_index]['start'] >= chr_start) & (peak_coor[chr_index]['end'] <= chr_end))))) and (len(set(seq_segment) - set('ACGT')) == 0) and abs(SeqUtils.GC(seq_segment) - GC_content) < 1:
                if len(seq_segment)==peak_length:
                    seqs.append('seq')
                    files.writelines(chr_index+'\t'+str(chr_start)+'\t'+str(chr_end)+'\n')
####################################################################################
def convert_data(index,flag='train'):
    train_data_list, table_name_list=get_data_name(data_names)
    narrowPeak='./inputs/TfbsUniform_hg19_ENCODE/'+table_name_list[index]+'.narrowPeak.gz'
    [peak_coor, peak_num] = get_peak_coor(narrowPeak)
    data_AC='./inputs/encode_101/'+train_data_list[index]+'_AC.seq.gz'
    data_B='./inputs/encode_101/'+train_data_list[index]+'_B.seq.gz'

    _=load_data_encode(data_AC,peak_coor,train_data_list[index], table_name_list[index], peak_flank=50)
    table_name = load_data_encode(data_B,peak_coor,train_data_list[index], table_name_list[index], peak_flank=50) 
    return table_name
if __name__=='__main__':
    for i in range(300,301):
        name=convert_data(index=i)
        getchrom_index(name)


