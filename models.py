import numpy as np   
import tensorflow as tf   
from layers import *

def masked_sigmoid_cross_entropy(preds, y, mask):

	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.expand_dims(mask,-1)
	y = tf.multiply(y, mask)
	preds = tf.multiply(preds, mask)
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=preds)
	return tf.reduce_mean(loss)

class dGCN():
	def __init__(self, name, placeholders, input_dim, options):
		self.input_dim = input_dim
		# self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
		lr = options['learning_rate']
		self.options = options
		self.name = name

		self.layers_0=[]
		self.layers_1=[]
		self.layers_2=[]

		self.vars_0 = {}
		self.vars_1 = {}
		self.activations_0 = []
		self.activations_1 = []
		self.activations_2 = []
		self.vars={}

		self.inputs_0 = placeholders['features_0']
		self.inputs_1 = placeholders['features_1']
		self.inputs_2 = placeholders['features_2']
		self.outputs = None

		self.loss = 0
		with tf.variable_scope(self.name + '_vars'):
			self.vars['combine'] = uniform([2, 1], name='combine')
		self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)
		# self.optimizer = tf.train.AdadeltaOptimizer(learning_rate = lr)
		
		self.opt_op = None
		self.build()

	def _build(self):
		self.layers_0.append(GraphConvolution(
			'GC00', 
			input_dim = self.input_dim,
			output_dim = self.options['hidden1'],
			placeholders = self.placeholders,
			act = tf.nn.relu,
			use_dropout=False,
			support_id = 0,
			sparse_inputs = True,
			bias = True,
			concat = False
			))

		self.layers_1.append(GraphConvolution(
			'GC10', 
			input_dim = self.input_dim,
			
			output_dim = self.options['hidden1'],
			placeholders = self.placeholders,
			act = tf.nn.relu,
			use_dropout=False,
			support_id = 1,
			sparse_inputs = True,
			bias = True,
			concat = True
			))		
		self.layers_2.append(GraphConvolution2(
			output_dim = 1,
			hidden_shape = self.options['hidden_shape'],
			hidden_dim = self.options['hidden1']*2,
			placeholders = self.placeholders,
			act = tf.nn.relu,
			use_dropout=False,
			support_id = 1,
			sparse_inputs = True,
			bias = True,
			concat = True
			))	


	def build(self):
		with tf.variable_scope(self.name):
			self._build()
		self.activations_0.append(self.inputs_0)
		self.activations_1.append(self.inputs_1)
		self.activations_2.append(self.inputs_2)

		for layer in self.layers_0:
			hidden = layer(self.activations_0[-1])
			self.activations_0.append(hidden)
		for layer in self.layers_1:
			hidden = layer(self.activations_1[-1])
			self.activations_1.append(hidden)
		tmp = tf.concat([self.activations_0[-1],self.activations_1[-1]], axis=-1)
		
		for layer in self.layers_2:
			hidden = layer(self.activations_2[-1],tmp)
			self.activations_2.append(hidden)

		self.debug = [self.activations_0[-2], self.placeholders['labels'], self.placeholders['labels_mask'], self.activations_0[-1]]
		self.outputs = tf.squeeze(input=self.activations_2[-1],axis=[0])

		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name:var for var in variables},

		# build metrics
		self._loss()
		self.opt_op = self.optimizer.minimize(self.loss)
	def _loss(self):
		self.loss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		self.preds = self.outputs
		self.labels = self.placeholders['labels']





