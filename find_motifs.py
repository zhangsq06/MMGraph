import os
import argparse
####################################################
parser = argparse.ArgumentParser(description="Finding motifs from ATAC-seq data.")
parser.add_argument('--in_path', default='./atac_results/fimotest',type=str,help='The prefix name of the bed file')
parser.add_argument('--out_path', default='./atac_results',type=str,help='Path to store motifs') 
args = parser.parse_args()
#############################################
def find_motifs(args):
    meme_files = "./hocomoco_meme"  #
    test_files = args.in_path   #Test data set
    meme = os.walk(meme_files)
    test = os.walk(test_files)
    os.system("rm -rf " +args.out_path +"/fimo_out/*")
    prefix = "fimo -o " + args.out_path +"/fimo_out/"  #Instruction prefix fimo -o and output path
    
    input1_list = []
    input2_list = []
    file_name_list = []
    file_name2_list = []
    ####################################
    for path,_,file_list in meme:    #The list of file names that make up the hocomoco dataset is easy to traverse later
        for file_name in file_list:
            input1 = os.path.join(path,file_name)
            input1_list.append(input1)
            file_name_list.append(file_name)
    ####################################
    for path2,_,file_list2 in test:   #The list of file names that make up test dataset is easy to traverse later
        for file_name2 in file_list2:
            input2 = os.path.join(path2,file_name2)
            input2_list.append(input2)
            file_name2_list.append(file_name2)
    ####################################
    for i in range(len(file_name_list)):    #Scan item by item
        file_name = file_name_list[i]
        input1 = input1_list[i]
        for j in range(len(file_name2_list)):
            file_name2 = file_name2_list[j]
            input2 = input2_list[j]
            cmd = prefix + file_name2[:-3] + file_name[:-5]+" " +input1 +" "+input2 #Combined execution command
            # try:
            os.system(cmd)
            # os.system("rm -rf fimo.txt")
            target = args.out_path+"/fimo_out/"+file_name2[:-3] + file_name[:-5]+"/fimo.gff"
            f = open(target,"rb").readlines()
            if len(f)<6:
                os.system("rm -rf " +target[:-9])
                # print("rm -rf "+target[:-9])
            # except:
            #     # print("error: "+file_name+" "+file_name2)
            #     pass
    ##########################################
    
    file_name = args.out_path + "/fasta_file/"   #Integrate information from .gff files and generate .fa files
    for root,dirs,files in os.walk(args.out_path + "/fimo_out"):
        tmp = root.split("/")
        if len(tmp) > 3:
            target =  root + "/fimo.gff"
            out = open(target,"r").readlines()[1:]
            data = ""
            for line in out:
                seq = line.split("\t")[0]
                res = line.split("\t")[-1].split(";")[-2].split("=")[-1]
                data += ">"+seq+"\n"+res+"\n"
            f = open(file_name + tmp[-1] + ".fa", "w")
            f.write(data)
            f.close()
###############################################
if __name__=="__main__":
    find_motifs(args)
