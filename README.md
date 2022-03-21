# GraphPred
 GraphPred is a tool based on GNNs and coexisting probability for predicting TFBSs and finding motifs from ATAC-seq data.
 
## Dependencies
Biopython-1.70, tensorflow-1.14.0, [meme-chip](https://meme-suite.org/meme/tools/meme-chip), [HINT](https://www.regulatory-genomics.org/hint/introduction/) and [TOBIAS](https://github.com/loosolab/TOBIAS)

## Step1 get your foorprints via the get_hint_tobias function, and return foorprints to the dir './tobias_footprint_bed/'
CMD: python get_hint_tobias --bed bed_name --path current_dir
Arguments --bed(the name of your file) --path ( current dir which contains hg38.fa GSE*.bed  and GSE*.bam)
## Step2 obtain the bed file of foorprints,
CMD python get_peaks --bed bed_name --path current_dir
This CMD need the bed file of original peaks as inputs and output the ,
Arguments --bed (the name of your file) --path(dir that contains the bed file of footprints)
## Step3 obtaining the fasta of trianing data and testing data,and return data to the dir './data/'
CMD python getchrom_index --bed bed_name --path dir --peak_flank 50
This CMD need the bed file of footprints as inputs,
Arguments --bed (the bed file of footprints) --path  (the dir  that includes the fasta of the footprint) --peak_flank (length of sequecnes in fasta file))
## Step4 generating adjacency matrixs
CMD python make_data.py
This CMD generate the adjacency matrix that contains similarity matrix, coexsiting matrix and inclusive matrix.
## Step5 training GraphPred model
cd ./GraphPred
CMD python train.py --dataset name
Arguments --dataset (the name of your data , such as GSE114204)
## Step6 predicting TFBSs from ATAC-seq data
CMD python get_tfbs.py --dataset name 
Arguments --dataset (the name of your data , such as GSE114204)
## Step7 finding multiple motifs from ATAC-seq data
CMD python find_motifs.py --dataset name
