# GraphPred
# Step1 get your foorprints via the get_hint_tobias function, and return foorprints to the dir './tobias_footprint_bed/'
CMD: python get_hint_tobias --bed bed_name --path current_dir
# Step2 obtain the bed file of foorprints,
CMD python get_peaks --bed bed_name --path current_dir
# Step3 obtaining the fasta of trianing data and testing data,and return data to the dir './data/'
CMD python getchrom_index --bed bed_name --path current_dir --peak_flank 50
# Step4 obtaining the fasta of trianing data and testing data,and return data to the dir './data/'

