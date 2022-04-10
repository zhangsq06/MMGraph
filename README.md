# GraphPred

 GraphPred is a tool based on graph neural network (GNN) and coexisting probability for predicting transcription factor bidning sites (TFBSs) and finding multiple motifs from ATAC-seq data.

## Dependencies

Python3.6,Biopython-1.70, tensorflow-1.14.0, [HINT](https://www.regulatory-genomics.org/hint/introduction/) and [TOBIAS](https://github.com/loosolab/TOBIAS)

## Data

Downloading bed and bam of GSE114204 from [the ENCODE project](https://www.encodeproject.org/).

## Step1 Getting foorprints via the get_hint_tobias function, and return foorprints to the dir './tobias_footprint_bed/'

CMD: python get_footprints.py --bed bed_name --path current_dir
Arguments --bed(the name of your file) --path ( current dir which contains hg38.fa GSE*.bed  and GSE*.bam)

## Step2 Obtain the bed file of foorprints

CMD python get_bed_footprints.py --bed bed_name --path current_dir
This CMD need the bed file of original peaks as inputs and output the ,
Arguments --bed (the name of your file) --path(dir that contains the bed file of footprints)

## Step3 Obtaining the fasta of trianing data and testing data,and return data to the dir './data/'

CMD python getchrom_index.py --bed bed_name --path dir --peak_flank 50
This CMD need the bed file of footprints as inputs,
Arguments --bed (the bed file of footprints) --path  (the dir  that includes the fasta of the footprint) --peak_flank (length of sequecnes in fasta file))

## Step4 Generating adjacency matrixs

CMD python cal_mat.py
This CMD generate the adjacency matrix that contains similarity matrix, coexsiting matrix and inclusive matrix.

## Step5 Training GraphPred model

cd ./GraphPred
CMD python train.py --dataset name
Arguments --dataset (the name of your data , such as GSE114204)

## Step6 Predicting TFBSs from ATAC-seq data

CMD python locate_tfbs.py --dataset name
Arguments --dataset (the name of your data , such as GSE114204)

# Step7 Merging TFBSs

CMD python merge_tfbs.py --dataset name
