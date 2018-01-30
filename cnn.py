import tensorflow as tf
import numpy as np

def cnn(reuse=None):

    input = tf.placeholder(tf.float32,shape = (None,12,8,8,1))
    z = [16,32]
    network = input
    print(network)   
    network = tf.layers.conv3d(input,8,3,name="Conv3d_1",reuse=None)
    print(network)
    

def get_network_endpoints():

    endpoints = {}
    x = tf.placeholder(tf.float32,shape = (None,height))
    keep_prob = tf.placeholder(tf.float32,shape = ())
    lr = tf.placeholder(tf.float32,shape = ())
    y = tf.placeholder(tf.float32,shape = (None,num_classes))
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

    return endpoints