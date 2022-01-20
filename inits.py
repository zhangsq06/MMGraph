import numpy as np  
import tensorflow as tf 



def normal(shape, name =None):
	n_input = shape[0]
	n_output = shape[1]
	tmp = tf.random_normal(shape, stddev = 0.01)
	return tf.Variable(tmp, name = name)

def uniform(shape, scale = 0.05, name=None):
	tmp = tf.random_uniform(shape, minval =-scale, maxval=scale, dtype=tf.float32)
	return tf.Variable(tmp, name=name)
def zeros(shape, name=None):
	tmp = tf.zeros(shape, dtype = tf.float32)
	return tf.Variable(tmp, name=name)
def ones(shape, name=None):
	tmp = tf.ones(shape, dtype = tf.float32)
	return tf.Variable(tmp, name=name)





