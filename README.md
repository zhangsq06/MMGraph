# MMGraph
 MMGraph is a multiple motif predictor based on graph neural network and coexisting probability for ATAC-seq data.
# Dependencies
Python3.6, Biopython-1.70, tensorflow-1.14.0, [bedtools v2.21.0](https://bedtools.readthedocs.io/en/latest/content/installation.html), [HINT](https://www.regulatory-genomics.org/hint/introduction/) and [TOBIAS](https://github.com/loosolab/TOBIAS).

# Data
GSE11420x: (290M) (https://drive.google.com/drive/folders/1KehQzVDcBE00wWdYcaq3P4LEAVHBK3ni?usp=sharing) 
hg38.fa: (3.05GB) (https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/) 
88 ATAC-seq datasets from the ENCODE project: (about 1TB) (https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false),
Put *.bed, *.bam, and hg38.fa into the dir 'test_data'.

# A simple tutorial 
## 1. Detecting foorprints via the TOBIAS and Hint-ATAC tools.
Usage: python detect_footprints.py --bed [prefix name]  
Arguments:   
--bed (the prefix name of your file).  

For example, if your input files are GSE11420x.bed and GSE11420x.bam, you only need to input GSE11420x)  --path (the dir contains hg38.fa, GSE11420x.bed and GSE11420x.bam)  

 Example:  
> python detect_footprints.py --bed GSE11420x  

Explain: The bed file of hg38.fa .bed  and .bam are required as inputs and outputs of the foorprints store in the file './test_data/tobias_top1500_bed/top1500GSE11420x.bed'.

# 2 Constructing the heterogeneous graph, training a GNN model and finding mutiple motifs
## 2.1 Constructing the heterogeneous graph (m=1024,n=3000).
Obtaining fixed 101 length of sequences (S) by the bedtools v2.21.0, which is used to construct the heterogeneous graph.
Usage: python get_S.py --bed [footprints file] --peak_flank [number]  

Arguments:  
--bed (footprints file, such as GSE11420x)  
--path  (the dir that includes the footprints and hg38.fa)  
--peak_flank (length of the sequence is equal to (peak_flank)*2+1)

Example:  
> python get_S.py --bed GSE11420x --peak_flank 50  

Explain: The bed file of footprints is required as inputs, and output DNA sequences are in './data/encode_101'.

Calculates three types of edges (similarity edges, coexsiting edges and inclusive edges) to construct the heterogeneous graph, where sequences and k-mers are nodes.  
Usage: python construct_graph.py --dataset [name] --k [k-mer]  
Arguments:   
--dataset (the name of sequence data, such as GSE11420x_encode)   
--k (the length of k-mer, default:k=5)
Example:  
>  python construct_graph.py --dataset GSE11420x_encode --k 5

## 2.2 Building and training a three-layer of GNN model to learn the embedding of nodes and identify whether a sequence contains TFBSs. 
Usage: python train.py --dataset [name]--k [k-mer]  
Arguments:   
--dataset (the name of sequence data, such as GSE11420x_encode)  
--k (the length of k-mer, default:k=5)  
Example:  
>  python train.py --dataset GSE11420x_encode --k 5  

Explain: This step trains a three-layer of GNN model on the GSE11420x_encodeGSE11420 dataset and save the embedding of nodes in './TFBSs/'.

## 2.3 Finding mutiple motifs with different lengths
We calcualtes the mutual information between the embedding of k-mer nodes and embedding of sequence nodes, and find multiple candidate TFBSs by the coexsiting probability between k-mer nodes.  
Usage: python find_multiple_tfbs.py --dataset [name]  
Arguments:   
--dataset (the name of sequence data, such as GSE11420x_encode)  

Example:
> python find_multiple_tfbs.py --dataset GSE11420x_encode  

Explain: The embeding of k-mers and sequences are required as inputs and outputs will be mutliple TFBSs with different lengths in ./test_data/motifs/GSE11420x_encode_0.81_merge_.fa.

