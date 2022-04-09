import tensorflow as tf  
from tqdm import tqdm
from cal_mat import GenerateAdjs
import scipy
from utils import *
import models
from models import  dGCN
import os

import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import argparse
################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--dataset', default='GSE114204_encode',type=str,help='The prefix name of the dataset')
parser.add_argument('--k', default=5,type=int,help='The length of K-mer')
args = parser.parse_args()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#######################################
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))
########################## the  length of k-mer
def LoadAdjs(tfids,Kmers):
#####calcualte the three types of weight matrix i.e. similarity matrix, coexisting matrix, inclusive matrix. 	
	sizes = [Kmers]
	ext = ".npz"
	strsize =""
	for size in sizes:
		strsize+=str(size)
	filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_inclu" + ext # inclusive edges
	if os.path.exists(filename):
		labelname = "./adjs/"+tfids + "_" + strsize + "_" + "labels.txt.npy" #labels
		labels = np.load(labelname)
		adj_inclu = scipy.sparse.load_npz(filename)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_sim" + ext #similarity edges
		adj_sim = scipy.sparse.load_npz(filename)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_coo" + ext #similarity edges
		adj_coo = scipy.sparse.load_npz(filename)
	else:
		adj_inclu, adj_sim, adj_coo, labels = GenerateAdjs(tfids,Kmers)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_inclu" + ext
		scipy.sparse.save_npz(filename, adj_inclu)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_sim" + ext
		scipy.sparse.save_npz(filename, adj_sim)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_coo" + ext
		scipy.sparse.save_npz(filename, adj_coo)
		labelname = "./adjs/"+tfids + "_" + strsize + "_" + "labels.txt"
		np.save(labelname,labels)
	return adj_inclu, adj_sim, adj_coo, labels
###########################################################	
def Task(name,Kmers,adjs_0, adjs_1, adjs_2, idx_train, idx_val, idx_test, idx_all, labels, options, save=False):
	n = len(labels)
	test_mask = np.zeros(n,dtype=np.int)
	val_mask = np.zeros(n,dtype=np.int)
	train_mask = np.zeros(n,dtype=np.int)

	train_mask[idx_train] = 1
	val_mask[idx_val] = 1
	test_mask[idx_test] = 1

	#define placeholders
	if options['model'] == 'gcn':
		support0 = [Process_adj(adj) for adj in adjs_0]
		support1 = [Process_adj(adj) for adj in adjs_1]
		num_support = len(adjs_0)
	features0 = scipy.sparse.identity(adjs_0[0].shape[0])
	features0 = Process_features(features0)
	features1 = scipy.sparse.identity(adjs_1[0].shape[0])
	features1 = Process_features(features1)
	features2 = adjs_2

	placeholders = {
		'support0':[tf.sparse_placeholder(tf.float32) for _ in range(num_support)],
		'support1':[tf.sparse_placeholder(tf.float32) for _ in range(num_support)],
		'features0':tf.sparse_placeholder(tf.float32, shape=None),
		'features1':tf.sparse_placeholder(tf.float32, shape=None),
		'features2':tf.placeholder(tf.float32, shape=None),
		'labels':tf.placeholder(tf.float32, shape=None),
		'labels_mask': tf.placeholder(tf.bool),
		'dropout': tf.placeholder_with_default(0., shape=()),
		'training': tf.placeholder_with_default(0., shape=())
	}
	
	# #build the model
	model = dGCN("gcn", placeholders, features_0[2][1], options)
	# #Initializing session
	sess = tf.Session(config=config)
	
	# # define model evaluation function
	def evaluate(features_0, features_1,features_2, support_0, support_1, label, mask, placeholders):
		feed_dict_val = construct_feed_dict(
			features_0, features_1,features_2, support_0, support_1, label, mask, placeholders)
		loss,preds,labels, _, _, seq_hidden, kmer_hidden = sess.run([model.loss, model.preds, model.labels, model.debug, model.activations_2, model.seq_hidden, model.kmer_hidden], feed_dict=feed_dict_val)
		return loss,preds,labels,seq_hidden, kmer_hidden
	sess.run(tf.global_variables_initializer())
	# train model
	feed_dict = construct_feed_dict(features0, features1, features2,support0, support1, labels, train_mask, placeholders)
	feed_dict.update({placeholders['dropout']:0.2})
	for epoch in tqdm(range(options['epochs']+1)):
		outs = sess.run([model.opt_op, model.loss, model.preds, model.debug, model.activations_2,model.seq_hidden, model.kmer_hidden], feed_dict=feed_dict)
		if epoch % 2 == 0:
			val_loss, preds, labels,_,_= evaluate(features0, features1,features2, support0, support1, labels, val_mask, placeholders)
			# print(preds)
			val_auc,fpr,tpr,thresholds = Com_auc(labels,preds,idx_val)
			print("epoch %d: valid loss = %f, val_auc = %f" %(epoch, val_loss, val_auc))
		if epoch == options['epochs']:
			test_loss, preds, labels,seq_hidden, kmer_hidden= evaluate(features0, features1,features2, support0, support1, labels, test_mask, placeholders)
			test_auc,fpr,tpr,thresholds = Com_auc(labels, preds,idx_test)
			print("epoch %d: test_auc = %f" %(epoch, test_auc))
			seq_path = './data/TFBS/'+name+str(Kmers)+'_seq'#sequence embedding
			np.save(seq_path, seq_hidden[-1])#sequence embedding
			kmer_path = './data/TFBS/'+name+str(Kmers)+'_kmer'
			np.save(kmer_path, kmer_hidden[-1])# save k-mer embedding
			out_test='./output/'+name+'_test.txt' ### testing results
			np.savetxt(out_test,[np.squeeze(labels[idx_test,:]), np.squeeze(preds[idx_test,:])])
			out_val='./output/'+name+'_val.txt' ### validation results
			np.savetxt(out_val,[np.squeeze(labels[idx_val,:]), np.squeeze(preds[idx_val,:])])
###################################################################################
def GraphPred(args):
	Kmers = int(args.kmer)
	# Settings
	options = {}
	options['model'] = 'gcn'
	options['epochs'] = 50
	options['dropout'] = 0.2
	options['weight_decay'] = 0.001
	options['hidden1'] = 150
	options['learning_rate'] = 0.001 
	########################
	tfid = args.dataset
	adj_inclu, adj_sim, adj_co, labels = LoadAdjs(tfid,Kmers)
	n = len(labels)
	#########################################
	adj1 = adj_co
	adj2 = adj_sim
	adj3 = adj_inclu.toarray()
	adjs_0=[adj1]
	adjs_1=[adj2]
	idx_all = np.array([i for i in range(n)],dtype='int')# indexs of all ATAC-seq sequences
    # splitting ATAC-seq sequences into training data, validation data and testing data.
	idx_train = idx_all[:int(0.8*n)]
	idx_val = idx_all[int(0.8*n):int(0.9*n)]
	idx_test = idx_all[int(0.9*n):]
	options['hidden_shape'] = adj1.toarray().shape
	Task(args.dataset,Kmers,adjs_0, adjs_1,adj3, idx_train, idx_val, idx_test, idx_all, labels, options, True)
##########################main##############
if __name__=='__main__':
	GraphPred(args)
