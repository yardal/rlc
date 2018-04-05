import tensorflow as tf
import numpy as np
import operator
import functools
import logging


def print_var_list():
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    total = 0
    for i in var_list:
        shape = i.get_shape().as_list()
        if len(shape) == 0:
            num_param_in_var = 1
        else:
            num_param_in_var = functools.reduce(operator.mul, shape)
        strr = i.name + "\tParams: " + str(num_param_in_var) + "\t" + str(i.get_shape().as_list())
        logging.info(strr.expandtabs(27))
        total = total + num_param_in_var
    logging.info("Total: " + str(total))


def cnn(input, reuse=None):
    z = [16, 32]
    network = input
    network = tf.layers.conv3d(network, 8, 3, 2, name="Conv3d_1", padding="SAME", reuse=reuse)
    network = tf.layers.conv3d(network, 16, 3, 2, name="Conv3d_2", padding="SAME", reuse=reuse)
    network = tf.layers.conv3d(network, 32, 3, 2, name="Conv3d_3", padding="SAME", reuse=reuse)

    shape = network.get_shape().as_list()
    dim = np.prod(shape[1:])

    network = tf.reshape(network, [-1, dim])
    network = tf.layers.dense(network, 16, tf.nn.relu)
    network = tf.layers.dense(network, 1)
    return network


class Network:

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 12, 8, 8, 1))
        self.keep_prob = tf.placeholder(tf.float32, shape=())
        self.lr = tf.placeholder(tf.float32, shape=())
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.net = cnn(self.x)
        self.prob = tf.nn.sigmoid(self.net)
        self.l2_lambda = 1e-5

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.net))
        l2_loss = 0
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            l2_loss = l2_loss + tf.norm(i, ord=1)

        loss = loss + self.l2_lambda * l2_loss
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def eval(self, tensor):

        return self.sess.run(self.prob, feed_dict={self.x: tensor})

