import tensorflow as tf
import numpy as np
import operator
import functools
import logging

def print_var_list():

	var_list =	tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

	total = 0
	for i in var_list:
		shape = i.get_shape().as_list()
		if len(shape) == 0:
			num_param_in_var = 1
		else:
			num_param_in_var = functools.reduce(operator.mul,shape)
		strr = i.name + "\tParams: " + str(num_param_in_var) + "\t" + str(i.get_shape().as_list())
		logging.info (strr.expandtabs(27))
		total = total + num_param_in_var
	logging.info("Total: " + str(total))
	
def cnn(input,reuse=None):

    z = [16,32]
    network = input
    network = tf.layers.conv3d(input,8,3,name="Conv3d_1",reuse=None)
    network = tf.layers.conv3d(network,16,3,name="Conv3d_2",reuse=None)
    network = tf.layers.conv3d(network,32,3,name="Conv3d_3",reuse=None)
    network = tf.layers.conv3d(network,32,2,name="Conv3d_4",reuse=None)
    return network

def get_network_endpoints():

   
    endpoints = {}
    x = tf.placeholder(tf.float32,shape = (None,12,8,8,1))
    keep_prob = tf.placeholder(tf.float32,shape = ())
    lr = tf.placeholder(tf.float32,shape = ())
    y = tf.placeholder(tf.float32,shape = (None,1))
    net = cnn(x,keep_prob)    
    prob = tf.nn.softmax(net)
    predict = tf.argmax(net,axis=1)
    
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = net))
    l2_loss = 0
    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        l2_loss = l2_loss + tf.norm(i,ord=1)
    
    loss = loss + 1e-5*l2_loss
    opt = tf.train.AdamOptimizer(lr).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(net,axis=1)),tf.float32))

    endpoints['x']         = x           
    endpoints['y']         = y           
    endpoints['net']       = net         
    endpoints['keep_prob'] = keep_prob   
    endpoints['lr']        = lr          
    endpoints['loss']      = loss        
    endpoints['prob']      = prob        
    endpoints['predict']   = predict     
    endpoints['opt']       = opt         
    endpoints['accuracy']  = accuracy    
	
    print_var_list()
    return endpoints
