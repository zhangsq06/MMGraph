# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
## compute aAUC
def Com_auc(test_l,pred,mask):
    ##real value，predicted value
	# mask=1>=mask
	test_l = test_l[mask,:]
	pred = pred[mask,:]
	fpr, tpr, thresholds = metrics.roc_curve(test_l, pred, pos_label=1)
	roc_auc = metrics.auc(fpr, tpr)
	return roc_auc,fpr,tpr,thresholds
##################################transper sparse matrix to tuple
def to_tuple(mx):
	if not sp.isspmatrix_coo(mx):
		mx = mx.tocoo()
	coords = np.stack((mx.row, mx.col)).transpose()
	values = mx.data
	shape = mx.shape
	return coords, values, shape
def Sparse_to_tuple(sparse_mx):
	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)
	return sparse_mx
#############################masking samples
def Sample_mask(idx, n):
	mask = np.zeros(n)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)
####### constructing dict to update hyper-parameter
def construct_feed_dict(features0, features1,features2, support0, support1, labels, labels_mask, placeholders):
	feed_dict = dict()
	feed_dict.update({placeholders['labels']:labels})
	feed_dict.update({placeholders['labels_mask']:labels_mask})
	feed_dict.update({placeholders['support0'][i]:support0[i] for i in range(len(support0))})
	feed_dict.update({placeholders['support1'][i]:support1[i] for i in range(len(support1))})
	feed_dict.update({placeholders['features0']:features0})
	feed_dict.update({placeholders['features1']:features1})
	feed_dict.update({placeholders['features2']:features2})
	return feed_dict
############################### normalizing adjacy matrix 
def Nor_adj(adj):
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def Process_features(features):
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum,-1).flatten()
	r_inv[np.isinf(r_inv)] = 0.0
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)
	return Sparse_to_tuple(features)

def Process_adj(adj):
	# print(adj.shape)
	adj_normalized = Nor_adj(sp.eye(adj.shape[0])+adj)
	return Sparse_to_tuple(adj_normalized)


