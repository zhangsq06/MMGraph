# GraphPred
 GraphPred is a tool to find multiple ATAC-seq motifs using graph neural network (GNN) and coexisting probability.
# Dependencies
Python3.6,Biopython-1.70, tensorflow-1.14.0, [bedtools v2.21.0](https://bedtools.readthedocs.io/en/latest/content/installation.html), [HINT](https://www.regulatory-genomics.org/hint/introduction/) and [TOBIAS](https://github.com/loosolab/TOBIAS).

# Data
Downloading [GSE11420x](https://drive.google.com/drive/folders/1KehQzVDcBE00wWdYcaq3P4LEAVHBK3ni?usp=sharing) and [hg38.fa](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/) .
Putting the *.bed, *.bam, and hg38.fa into the dir 'test_data'.


# A simple tutorial 
## 1. Detecting foorprints via the TOBIAS and Hint-ATAC tools.
CMD: python detect_footprints.py --bed GSE11420x
This CMD needs the bed file of hg38.fa .bed  and .bam as inputs and outputs the foorprints in the file './test_data/tobias_top1500_bed/top1500GSE11420x.bed'.
Arguments --bed (the prefix name of your file. For example, if your input files are GSE11420x.bed and GSE11420x.bam, you only need to input GSE11420x) --path (the dir contains hg38.fa, GSE11420x.bed and GSE11420x.bam)


# 2 Constructing the heterogeneous graph, training a GNN model and finding mutiple motifs
## 2.1 Constructing the heterogeneous graph.
Obtaining fixed 101 length of sequences by the bedtools v2.21.0, which is used to construct the heterogeneous graph.
CMD python get_S.py --bed GSE11420x --peak_flank 50
This CMD needs the bed file of footprints as inputs, and output sequences in the './data/encode_101'.
Arguments --bed (the bed file of footprints , such as GSE11420x) --path  (the dir that includes the footprints and hg38.fa) --peak_flank (length of the sequence is equal to (peak_flank)*2+1)

CMD python construct_graph.py --dataset GSE11420x_encode --k 5
This CMD calculates three types of edges (similarity edges, coexsiting edges and inclusive edges) to the heterogeneous graph, where sequences and k-mers are nodes.
Arguments --dataset (the name of sequence data, such as GSE11420x_encode) --k (the length of k-mer, default:k=5)


## 2.2 Building and training a three-layer of GNN model to learn the embedding of nodes and identify whether a sequence contains TFBSs. 
CMD python train.py --dataset GSE11420x_encode --k 5
This CMD trains a three-layer of GNN model on the GSE11420x_encodeGSE11420 dataset and save the embedding of nodes in './TFBSs/'.

## 2.3 Finding mutiple motifs
We calcualtes the mutual information between the embedding of k-mer nodes and embedding of sequence nodes, and find multiple candidate TFBSs by the coexsiting probability between k-mer nodes.
CMD python find_tfbs.py --dataset GSE11420x_encode
This CMD needs the embeding of k-mers and sequences as inputs and outputs candidate TFBSs in ./test_data/motifs/GSE11420x_encode_*_.fa.
IF mulitple candidate TFBSs have overlap, they may be further merged to a longer TFBS.
CMD python merge_tfbs.py --tfbs ./test_data/motifs/GSE11420x_encode_0.77_.fa
This CMD needs the candidate TFBSs as input and ouputs the mergered TFBSs.
Arguments --tfbs (the name of the tfbs seq , such as ./test_data/motifs/GSE11420x_encode_0.81_.fa)