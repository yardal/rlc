import tensorflow as tf
import numpy as np
import operator
import functools
import logging
import os
import random


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


def cnn(input, path):
    with tf.variable_scope(path):
        network = input
        network = tf.layers.conv3d(network, 8, 3, 2, name="Conv3d_1", padding="SAME")
        network = tf.layers.conv3d(network, 16, 3, 2, name="Conv3d_2", padding="SAME")
        network = tf.layers.conv3d(network, 32, 3, 2, name="Conv3d_3", padding="SAME")

        shape = network.get_shape().as_list()
        dim = np.prod(shape[1:])

        network = tf.reshape(network, [-1, dim])
        network = tf.layers.dense(network, 16, tf.nn.relu)
        network = tf.layers.dense(network, 1)
        return network


class Network:

    def __init__(self, path):

        with tf.variable_scope(path):
            self.x = tf.placeholder(tf.float32, shape=(None, 12, 8, 8, 1), name="input")
            self.y = tf.placeholder(tf.float32, shape=(None, 1), name="output")
            self.lr = 1e-4
            self.net = cnn(self.x,path)
            self.prob = tf.nn.sigmoid(self.net)
            self.l2_lambda = 1e-9

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.net))
            l2_loss = 0
            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                l2_loss = l2_loss + tf.norm(i, ord=1)

            loss = loss + self.l2_lambda * l2_loss
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

            self.sess = tf.Session()

            self.saver = tf.train.Saver()
            self.model_path = os.path.join(path + '/model.ckpt')
            try:
                self.saver.restore(self.sess, self.model_path)

            except tf.errors.InvalidArgumentError:
                self.sess.run(tf.global_variables_initializer())
                self.saver.save(self.sess, self.model_path)
                print("Initializing network")

    def eval(self, tensor):

        return self.sess.run(self.prob, feed_dict={self.x: tensor})

    def train(self):

        labels_dict = {"1.0.npy": 1.0, "0.0.npy": 0.0, "0.5.npy": 0.5}
        batch_size = 1
        files = os.listdir("data")
        for j in range(1000):

            random.shuffle(files)
            batch = np.zeros((batch_size, 12, 8, 8, 1))
            labels = np.zeros((batch_size, 1))
            for i in range(batch_size):
                file_name = files[i]
                suffix = (file_name.split("_")[1])
                labels[i] = labels_dict[suffix]
                batch[i] = np.load(os.path.join("data", file_name))

            self.sess.run(self.opt, feed_dict={self.y: labels, self.x: batch})
        self.saver.save(self.sess, self.model_path)