import tensorflow as tf
from inits import *

def dot(x, y, sparse=True):
	if sparse:
		res = tf.sparse_tensor_dense_matmul(x, y)
	else:
		res = tf.matmul(x, y)
	return res
def sparse_dropout(x, keep_prob):
	keep_tensor = keep_prob + tf.random_uniform(tf.shape(x))
	drop_mask = tf.cast(tf.floor(keep_tensor), dtype=tf.bool)
	out = tf.sparse_retain(x, drop_mask)
	return out * (1.0/keep_prob)
#############the first lyaer of GNN#############
class GraphConvolution():
	def __init__(self, name, input_dim, output_dim, placeholders, use_dropout=True,
		support_id = None, sparse_inputs=False, act = tf.nn.relu, bias=True, concat = True):
		self.act = act
		self.sparse_inputs = sparse_inputs
		self.concat = concat

		self.debug = None
		self.placeholders = placeholders
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = 0.00001
		else:
			self.dropout = tf.constant(0.0)
		self.name = name
		self.bias = bias
		if support_id is None:
			self.support = placeholders['support0']
		else:
			self.support = placeholders['support' + str(support_id)]
		self.vars={}
		with tf.variable_scope(self.name +'_vars'):
			for i in range(len(self.support)):
				if concat:
					tmp = int(output_dim/(1.0 * len(self.support)))
				else:
					tmp = output_dim
				self.vars['weights_'+str(i)] = normal([input_dim, tmp], name='weights_' + str(i))
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')

	def __call__(self, inputs):
		x = inputs
		if self.use_dropout:			
			x = sparse_dropout(x, 1-self.dropout)

		outputs=[]
		for i in range(len(self.support)):
			pre_sup = dot(x, self.vars['weights_'+str(i)],sparse=True)
			output = dot(self.support[i], pre_sup,sparse=True)
			outputs.append(output)
		if self.concat:
			outputs=tf.concat(outputs,axis=-1)
		else:
			outputs=tf.add_n(outputs)/(1.0 * len(self.support))
		
		if self.bias:
			outputs+=self.vars['bias']
		return self.act(outputs)
##############the second layer of GNN#######################
class GraphConvolution2():
	def __init__(self, placeholders,hidden_shape=[1024, 1024],hidden_dim=800,output_dim=1,name='GCN', use_dropout=False,
		support_id = None, sparse_inputs=False, act = tf.nn.relu, bias=True, concat = True):
		self.act = act
		self.sparse_inputs = sparse_inputs
		self.concat = concat

		self.debug = None
		self.placeholders = placeholders
		self.dropout = placeholders['dropout']
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = placeholders['dropout']
		else:
			self.dropout = tf.constant(0.0)
		self.name = name
		self.bias = bias

		self.vars={}
		
		with tf.variable_scope(self.name +'_vars'):
			self.vars['weights_'+str(20)] = normal(hidden_shape, name='weights_' + str(20))
			self.vars['weights_'+str(21)] = normal([hidden_dim, output_dim], name='weights_' + str(21))
			self.vars['weights_'+str(22)] = normal([hidden_dim, hidden_dim], name='weights_' + str(22))
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')

	def __call__(self, input1,input2):
		x1 = input1
		x2 = input2#1024*800
		if self.use_dropout:
			x = tf.nn.dropout(x1, 1 - self.dropout)
		outputs=[]
		### transfer the x1,x2 to the same space
		# trans_m = dot(x2,self.vars['weights_'+str(22)], sparse=False)#1024*800
		pre_sup = tf.matmul(x1, self.vars['weights_'+str(20)], b_is_sparse=False)#1996*1024
		
		pre_sup = dot(pre_sup, x2,sparse=False)#1996*800
		output = dot(pre_sup,self.vars['weights_'+str(21)],sparse=False)
		outputs.append(output)
		
		if self.bias:
			outputs+=self.vars['bias']
		return self.act(outputs),pre_sup # OUTPU AND embedding


