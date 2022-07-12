import os
import argparse
#If multiple tfbs overlapped, they may be merged to a longer TFBS
#CMD python merge_tfbs.py --tfbs ./test_data/motifs/GSE11420x_encode_0.77_.fa
################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--tfbs', default='test_data/motifs/GSE114204_encode_0.8_.fa',type=str,help='TFBSs')
args = parser.parse_args()
##################
def Merges(name):
    FileR = open(name,"r")
    # if not os.path.exists("out"):
    #     os.makedirs("out")
    out_name = name[:-3]+'_merge.fa'
    print(out_name)
    FileW = open(out_name,"w+")
    done = 0
    x = FileR.readline()
    y = FileR.readline()
    i = int(x.split("_")[1])
    start = int(x.split("_")[2])
    end = int(x.split("_")[3])
    temp = y
    while(not done):
        x = FileR.readline()
        y = FileR.readline()
        if(x != ''):
            if(int(x.split("_")[1]) == i):
                if(int(end - int(x.split("_")[2])) >= 0):
                    cha = end - int(x.split("_")[2])
                    temp = temp.strip() + y[cha:]
                    end = int(x.split("_")[3])
                else:
                    FileW.write(">seq_"+ str(i) +"_" + str(start) + "_" + str(end) + "\n")
                    FileW.write(temp.strip() + "\n")
                    start = int(x.split("_")[2])
                    end = int(x.split("_")[3])
                    temp = y
            else:
                FileW.write(">seq_"+ str(i) +"_" + str(start) + "_" + str(end) + "\n")
                FileW.write(temp.strip() + "\n")
                i = int(x.split("_")[1])
                start = int(x.split("_")[2])
                end = int(x.split("_")[3])
                temp = y
        else:
            done = 1
    FileR.close()
    FileW.close()
if __name__ == '__main__':
    Merges(name=args.tfbs)