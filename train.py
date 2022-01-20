import tensorflow as tf  
import sys
from tqdm import tqdm
from make_data import generateAdjs
import scipy
from utils import *
import models
from models import  dGCN
from sklearn import metrics
import os
import time
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#######################################
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))
##########################

def loadAdjs(tfids,Kmers):

	print("PROCESSING %s" % tfid)
	
	sizes = [Kmers]
	ext = ".npz"
	strsize =""
	for size in sizes:
		strsize+=str(size)
	filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_inclu" + ext
	if os.path.exists(filename):
		labelname = "./adjs/"+tfids + "_" + strsize + "_" + "labels.txt.npy"
		labels = np.load(labelname)
		adj_inclu = scipy.sparse.load_npz(filename)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_sim" + ext
		adj_sim = scipy.sparse.load_npz(filename)
		filename = "./adjs/"+tfids + "_" + strsize + "_" + "adj_coo" + ext
		adj_coo = scipy.sparse.load_npz(filename)
	else:
		adj_inclu, adj_sim, adj_coo, labels = generateAdjs(tfids,Kmers)
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
def one_task(adjs_0, adjs_1, adjs_2, idx_train, idx_val, idx_test, idx_all, labels, options, save=False):
	n = len(labels)
	start_time = time.time()
	test_mask = np.zeros(n,dtype=np.int)
	val_mask = np.zeros(n,dtype=np.int)
	train_mask = np.zeros(n,dtype=np.int)

	train_mask[idx_train] = 1
	val_mask[idx_val] = 1
	test_mask[idx_test] = 1

	#define placeholders
	if options['model'] == 'gcn':
		support_0 = [preprocess_adj(adj) for adj in adjs_0]
		support_1 = [preprocess_adj(adj) for adj in adjs_1]
		num_support = len(adjs_0)
	features_0 = scipy.sparse.identity(adjs_0[0].shape[0])
	features_0 = preprocess_features(features_0)
	features_1 = scipy.sparse.identity(adjs_1[0].shape[0])
	features_1 = preprocess_features(features_1)
	features_2 = adjs_2

	placeholders = {
		'support_0':[tf.sparse_placeholder(tf.float32) for _ in range(num_support)],
		'support_1':[tf.sparse_placeholder(tf.float32) for _ in range(num_support)],
		#'features':tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
		'features_0':tf.sparse_placeholder(tf.float32, shape=None),
		'features_1':tf.sparse_placeholder(tf.float32, shape=None),
		'features_2':tf.placeholder(tf.float32, shape=None),
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
		loss,preds,labels,_,_ = sess.run([model.loss, model.preds, model.labels, model.debug, model.activations_2], feed_dict=feed_dict_val)
		return loss,preds,labels

	sess.run(tf.global_variables_initializer())
	cost_val=[]
	# train model
	feed_dict = construct_feed_dict(features_0, features_1, features_2,support_0, support_1, labels, train_mask, placeholders)
	feed_dict.update({placeholders['dropout']:0.2})
	for epoch in tqdm(range(options['epochs']+1)):
		outs = sess.run([model.opt_op, model.loss, model.preds, model.debug, model.activations_2], feed_dict=feed_dict)
		# print("epoch %d: trn lost = %f" %(epoch, outs[1]))
		if epoch % 2 == 0:
			val_loss, preds, labels= evaluate(features_0, features_1,features_2, support_0, support_1, labels, val_mask, placeholders)
			val_auc,fpr,tpr,thresholds = com_auc(labels,preds,idx_val)
			print("epoch %d: valid loss = %f, val_auc = %f" %(epoch, val_loss, val_auc))
		if epoch == options['epochs']:
			test_loss, preds, labels= evaluate(features_0, features_1,features_2, support_0, support_1, labels, test_mask, placeholders)
			test_auc,fpr,tpr,thresholds = com_auc(labels, preds,idx_test)
			print("epoch %d: test_auc = %f" %(epoch, test_auc))
###################################################################################
dataset = sys.argv[1]
Kmers = int(sys.argv[2])
start_time = time.time()
# Settings
options = {}
options['model'] = 'gcn'
options['epochs'] = 50
options['dropout'] = 0.2
options['weight_decay'] = 0.001
options['hidden1'] = 150
options['learning_rate'] = 0.001 

tfid = dataset
local_weights = [0.1]
adj_inclu, adj_sim, adj_coo, labels = loadAdjs(tfid,Kmers)
n = len(labels)
#########################################
aucs =[]
adj1 = 1* adj_coo # 5.0
adj2 = adj_sim
adj3 = adj_inclu.toarray()
print(adj1.toarray().shape,adj2.toarray().shape, adj3.shape)
adjs_0=[adj1]
adjs_1=[adj2]
idx_all = np.array([i for i in range(n)],dtype='int')
idx_train = idx_all[:int(0.8*n)]
idx_val = idx_all[int(0.8*n):int(0.9*n)]
idx_test = idx_all[int(0.9*n):]
options['hidden_shape'] = adj1.toarray().shape
one_task(adjs_0, adjs_1,adj3, idx_train, idx_val, idx_test, idx_all, labels, options, True)
# metric,embed0,embed1 = one_task(seqs, adjs_0, adjs_1, idx_train, idx_val, idx_test, idx_all, labels, options, True)
# print(metric,embed0[2].shape,embed1[2].shape)