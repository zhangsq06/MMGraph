import os
import argparse
################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--tfbs', default='GSE114204_encode.fa',type=str,help='TFBSs')
args = parser.parse_args()
##################
def Merges(name):
    FileR = open(name,"r")
    # if not os.path.exists("out"):
    #     os.makedirs("out")
    out_name = name.split('.')[0]+'_merge.fa'
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
                if(int(x.split("_")[2]) - 1 <= end):
                    cha = end - int(x.split("_")[2])
                    temp = temp[:-1] + y[(cha + 2):]
                    end = int(x.split("_")[3])
                else:
                    FileW.write(">seq_"+ str(i) +"_" + str(start) + "_" + str(end) + "\n")
                    FileW.write(temp)
                    start = int(x.split("_")[2])
                    end = int(x.split("_")[3])
                    temp = y
            else:
                FileW.write(">seq_"+ str(i) +"_" + str(start) + "_" + str(end) + "\n")
                FileW.write(temp)
                i = int(x.split("_")[1])
                start = int(x.split("_")[2])
                end = int(x.split("_")[3])
                temp = y
        else:
            done = 1
    FileR.close()
    FileW.close()
if __name__ == '__main__':
    Merges(args.tfbs)