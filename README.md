# GraphPred
 GraphPred is a tool based on GNNs and coexisting probability for predicting TFBSs and finding motifs from ATAC-seq data.
 
## Dependencies
Biopython-1.70, tensorflow-1.14.0, [meme-chip](https://meme-suite.org/meme/tools/meme-chip), [HINT](https://www.regulatory-genomics.org/hint/introduction/) and [TOBIAS](https://github.com/loosolab/TOBIAS)
## Step1 get your foorprints via the get_hint_tobias function, and return foorprints to the dir './tobias_footprint_bed/'
CMD: python get_hint_tobias --bed bed_name --path current_dir
## Step2 obtain the bed file of foorprints,
CMD python get_peaks --bed bed_name --path current_dir
## Step3 obtaining the fasta of trianing data and testing data,and return data to the dir './data/'
CMD python getchrom_index --bed bed_name --path current_dir --peak_flank 50

## Step4 generate the generate adjacency matrix
CMD python python make_data.py
