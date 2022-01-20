import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import pandas as pd
import seaborn as sns
import re
from matplotlib import pyplot as plt
from sklearn import metrics

def com_auc(test_l,pred,mask):
    ## 真实值，预测值
	# mask=1>=mask
	test_l = test_l[mask,:]
	pred = pred[mask,:]
	fpr, tpr, thresholds = metrics.roc_curve(test_l, pred, pos_label=1)
	roc_auc = metrics.auc(fpr, tpr)
	return roc_auc,fpr,tpr,thresholds




def summarize_results_over_celllines(filename):
	data = pd.read_excel(filename)
	df = pd.DataFrame(data,columns=['name','DB_star','DB','MEME1','MEME2','proposed'])
	names = df['name'].tolist()
	meme1 = df['MEME1'].tolist()
	meme2 = df['MEME2'].tolist()
	db= df['DB'].tolist()
	gb = df['proposed'].tolist()

	cellines2meme1 = {}
	cellines2meme2={}
	cellines2db={}
	cellines2gb={}

	for i, name in enumerate(names):
		auc_db = db[i]
		auc_meme1 = meme1[i]
		auc_meme2 = meme2[i]
		auc_gb = gb[i]
		# process name
		#print(name)
		name = name.strip()
		words = name.split('_')
		celline = words[1]
		if celline == "A549":
			print(name)
		
		if celline not in cellines2db:
			cellines2db[celline] = [auc_db]
		else:
			tmp = cellines2db[celline]
			tmp.append(auc_db)
			cellines2db[celline] = tmp
		if celline not in cellines2gb:
			cellines2gb[celline] = [auc_gb]
		else:
			tmp = cellines2gb[celline]
			tmp.append(auc_gb)
			cellines2gb[celline] = tmp
		if celline not in cellines2meme1:
			cellines2meme1[celline] = [auc_meme1]
		else:
			tmp = cellines2meme1[celline]
			tmp.append(auc_meme1)
			cellines2meme1[celline] = tmp
		if celline not in cellines2meme2:
			cellines2meme2[celline] = [auc_meme2]
		else:
			tmp = cellines2meme2[celline]
			tmp.append(auc_meme2)
			cellines2meme2[celline] = tmp


	meme1_auc = []
	meme2_auc = []
	db_auc = []
	gb_auc = []
	for cell in cellines2db:
		n_exp = len(cellines2meme1[cell])
		if n_exp <= 3:
			continue
		if cell != 'HepG2':
			continue
		print(cellines2db[cell])
		print(cellines2gb[cell])
		print("----------------------------")

		meme1_auc.append(np.mean(cellines2meme1[cell]))
		meme2_auc.append(np.mean(cellines2meme2[cell]))
		db_auc.append(np.mean(cellines2db[cell]))
		gb_auc.append(np.mean(cellines2gb[cell]))

		print(n_exp, cell, np.mean(cellines2meme1[cell]), np.mean(cellines2meme2[cell]), np.mean(cellines2db[cell]),np.mean(cellines2gb[cell]))
		print(np.std(cellines2meme1[cell]), np.std(cellines2meme2[cell]), np.std(cellines2db[cell]),np.std(cellines2gb[cell]))

	print(np.mean(meme1_auc),np.mean(meme2_auc),np.mean(db_auc),np.mean(gb_auc))
		

def summarize_results_over_TF(filename):
	data = pd.read_excel(filename)
	df = pd.DataFrame(data,columns=['name','DB_star','DB','MEME1','MEME2','proposed'])
	names = df['name'].tolist()
	meme1 = df['MEME1'].tolist()
	meme2 = df['MEME2'].tolist()
	db= df['DB'].tolist()
	gb = df['proposed'].tolist()


	print(len(names),len(meme1),len(meme2), len(db), len(gb))
	print(np.mean(meme1), np.mean(meme2), np.mean(db), np.mean(gb))

	cellines2meme1 = {}
	cellines2meme2={}
	cellines2db={}
	cellines2gb={}

	for i, name in enumerate(names):
		auc_db = db[i]
		auc_meme1 = meme1[i]
		auc_meme2 = meme2[i]
		auc_gb = gb[i]
		# process name
		name = name.strip()
		words = name.split('_')
		celline = words[0]
		#print(celline)
		if celline not in cellines2db:
			cellines2db[celline] = [auc_db]
		else:
			tmp = cellines2db[celline]
			tmp.append(auc_db)
			cellines2db[celline] = tmp
		if celline not in cellines2gb:
			cellines2gb[celline] = [auc_gb]
		else:
			tmp = cellines2gb[celline]
			tmp.append(auc_gb)
			cellines2gb[celline] = tmp
		if celline not in cellines2meme1:
			cellines2meme1[celline] = [auc_meme1]
		else:
			tmp = cellines2meme1[celline]
			tmp.append(auc_meme1)
			cellines2meme1[celline] = tmp
		if celline not in cellines2meme2:
			cellines2meme2[celline] = [auc_meme2]
		else:
			tmp = cellines2meme2[celline]
			tmp.append(auc_meme2)
			cellines2meme2[celline] = tmp


	meme1_auc = []
	meme2_auc = []
	db_auc = []
	gb_auc = []
	for tf in cellines2db:
		meme1_auc.append(np.mean(cellines2meme1[tf]))
		meme2_auc.append(np.mean(cellines2meme2[tf]))
		db_auc.append(np.mean(cellines2db[tf]))
		gb_auc.append(np.mean(cellines2gb[tf]))
	n = len(meme1_auc)
	a = pd.DataFrame({ '' : np.repeat('MEME-SUM',n), 'mean AUC': np.array(meme1_auc) })
	b = pd.DataFrame({ '' : np.repeat('MEME-M1',n), 'mean AUC': np.array(meme2_auc) })
	c = pd.DataFrame({ '' : np.repeat('DeepBind',n), 'mean AUC': np.array(db_auc) })
	d = pd.DataFrame({ '' : np.repeat('GraphBind',n), 'mean AUC': np.array(gb_auc) })

	df=a.append(b).append(c).append(d)
	ax = sns.boxplot(x="", y='mean AUC', data=df)
	medians = df.groupby([''])['mean AUC'].median()
	medians =[medians['MEME-SUM'], medians['MEME-M1'], medians['DeepBind'], medians['GraphBind']]
	nobs = df[''].value_counts().values
	nobs = [str(x) for x in nobs.tolist()]

	pos = range(len(nobs))
	print(ax.get_xticklabels())
	ax.set_title('ChIP models (506 experiments)')
	for tick, label in zip(pos, ax.get_xticklabels()):
		ax.text(pos[tick],medians[tick] + 0.005, "%4f" % medians[tick],horizontalalignment='center', size='medium', color='black')
	print(nobs)
	print(medians)

	#plt.show()
	return

	print(len(gb_auc))
	data = np.array([meme1_auc, meme2_auc, db_auc, gb_auc])
	data=data.T
	data = data.tolist()
	df = pd.DataFrame(data, columns=['MEME-SUM','MEME-M1','DeepBind','GraphBind'])

	ax = sns.boxplot(x="A",y="B",data=df)
	#median = df.groupby(['methods'])['mean AUC'].median().values
	#print(median)
	#plt.show()
	#exit(1)
	return

	meme1_mean = np.mean(meme1_auc)
	meme2_mean = np.mean(meme2_auc)
	db_mean = np.mean(db_auc)
	gb_mean = np.mean(gb_auc)

	meme1_std = np.std(meme1_auc)
	meme2_std = np.std(meme2_auc)
	db_std = np.std(db_auc)
	gb_std = np.std(gb_auc)

	print(meme1_mean, meme1_std)
	print(meme2_mean, meme2_std)
	print(db_mean, db_std)
	print(gb_mean, gb_std)

	exit(1)
	minvalue = 0
	bars = ['MEME-SUM', 'MEME-M1', 'DeepBind', 'GraphBind']
	means = [meme1_mean - minvalue, meme2_mean - minvalue, db_mean- minvalue, gb_mean- minvalue]
	stds = [meme1_std, meme2_std, db_std, gb_std]
	N=4
	ind = np.arange(N)
	width =0.35
	fig, ax = plt.subplots()
	rect = ax.bar(ind, means, width, color=['yellow','blue', 'green','red'],yerr=stds)
	ax.set_ylabel('ChIP AUC')
	ax.set_title('ChIP models (506 experiments)')
	ax.set_xticks(ind)
	ax.set_xticklabels(('MEME-SUM','MEME-M1','DeepBind','GraphBind'))
	ax.set_ylim((0.4, 1.0))
	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/1., 1.01*height,
                '%.4f' % float(height+minvalue),
                ha='center', va='bottom')
	autolabel(rect)
	plt.show()


	#tmp1=[]
	#tmp2=[]
	#for celline in cellines2db:
	#	n_exp = len(cellines2db[celline])
		
	#	if n_exp < 9:
	#		continue
		#print(celline, cellines2db[celline],cellines2proposed[celline])
	#	avg_db = np.mean(cellines2db[celline])
	#	avg_proposed = np.mean(cellines2proposed[celline])
	#	print(celline, avg_db,  avg_proposed, n_exp)
	#	tmp1.append(avg_db)
	#	tmp2.append(avg_proposed)

	#print(np.mean(tmp1),np.mean(tmp2))
	#print("number of celllines=%d" % len(tmp1))
	#print(cellines2proposed['HepG2'])




# summarize_results_over_TF("results.xlsx")


def get_angles(pos, i, d_model):
	angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(d_model))
	return pos * angle_rates

def positional_encoding(position, d_model):
	tmp1 = np.arange(position)
	tmp1 = tmp1[:, np.newaxis]
	tmp2 = np.arange(d_model)
	tmp2 = tmp2[np.newaxis, :]

	angle_rads = get_angles(tmp1, tmp2, d_model)
	sines = np.sin(angle_rads[:, 0::2])
	cosines = np.cos(angle_rads[:, 1::2])

	pos_encoding = np.concatenate([sines, cosines], axis=-1)
	return pos_encoding


def sample_mask(idx, n):
	mask = np.zeros(n)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)

def construct_feed_dict(features_0, features_1,features_2, support_0, support_1, labels, labels_mask, placeholders):
	feed_dict = dict()
	feed_dict.update({placeholders['labels']:labels})
	feed_dict.update({placeholders['labels_mask']:labels_mask})
	feed_dict.update({placeholders['support_0'][i]:support_0[i] for i in range(len(support_0))})
	feed_dict.update({placeholders['support_1'][i]:support_1[i] for i in range(len(support_1))})
	feed_dict.update({placeholders['features_0']:features_0})
	feed_dict.update({placeholders['features_1']:features_1})
	feed_dict.update({placeholders['features_2']:features_2})
	return feed_dict


def sparse_to_tuple(sparse_mx):
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.stack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		#tmp = []
		#tmp.append(shape[0])
		#tmp.append(shape[1])
		#shape = np.array(tmp, dtype=np.int64)
		
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)
	return sparse_mx


def normalize_adj(adj):
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_features(features):
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum,-1).flatten()
	r_inv[np.isinf(r_inv)] = 0.0
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)
	return sparse_to_tuple(features)

def preprocess_adj(adj):
	# print(adj.shape)
	adj_normalized = normalize_adj(sp.eye(adj.shape[0])+adj)
	return sparse_to_tuple(adj_normalized)

def chebyshev_polynomials(adj, k):
	adj_normalized = normalize_adj(adj)
	laplacian = sp.eye(adj.shape[0]) - adj_normalized
	largest_eigval, _ = eigsh(laplacian, 1, which='LM')
	scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

	t_k = list()
	t_k.append(sp.eye(adj.shape[0]))
	t_k.append(scaled_laplacian)

	def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_laplacian):
		s_lap = sp.csr_matrix(scaled_laplacian, copy=True)
		return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two
	for k in range(2, k+1):
		t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
	return sparse_to_tuple(t_k)





#posenc = positional_encoding(position, d_model)


