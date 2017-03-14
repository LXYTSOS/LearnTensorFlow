# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:50:20 2017

@author: sl169
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#xavier初始化器，能够根据某一层网络的输入、输出节点数量自动调整最合适的分布
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        #为输入x创建一个维度为n_input的placeholder
        self.x=tf.placeholder(tf.float32, [None, self.n_input])
        #建立一个能提取特征的隐藏层，然后对x加上噪声，与权重w1相乘，加上偏置b1
        #最后使用transfer对结果进行激活函数处理
        self.hidden = self.transfer(tf.add(tf.matmul(
                self.x + scale * tf.random_normal((n_input,)),
                self.weights['w1']), self.weights['b1']))
        #经过隐藏层后，需要在输出层进行数据复原、重建操作，这里就不需要激活函数了，直接乘上权重加上偏置
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']),self.weights['b2'])
        
        #接下来定义自编码器的损失函数，这里使用平方误差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                self.reconstruction, self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                   self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                   dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                   self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                   dtype = tf.float32))
        return all_weights
        
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
            feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost
        
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,
            self.scale: self.training_scale})
    
#    返回自编码器隐藏层的输出结果，提供一个接口来获取抽象后的特征，自编码器的隐藏层的
#    主要功能就是学习出数据中的高阶特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,
            self.scale: self.training_scale})
    
#    将隐藏层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict = {self.hidden: hidden})
    
#    整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据，即包括transform和
#    generate两部分，输入数据是原始数据，输出数据是复原后的数据。
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
            self.scale: self.training_scale})
    
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

#使用sklearn.preprocessing的StandardScaler，现在训练集上fit，再将这个Scaler用到训练
#数据和测试数据上，必须保证训练数据和测试数据都使用完全相同的Scaler，
#这样才能保证后面模型处理数据时的一致性。
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

#随机获取block数据的函数
def get_random_block_form_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

mnist = input_data.read_data_sets("../MNIST/MNIST_data/", one_hot=True)

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_stop = 1
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
        n_hidden = 200,
        transfer_function = tf.nn.softplus,
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
        scale = 0.01)
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_form_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples + batch_size
    if epoch % display_stop == 0:
        print("Epoch:",'%04d' % (epoch +1), "cost=", "{:.9f}".format(avg_cost))
total_cost = int(autoencoder.calc_total_cost(X_test))
print("Total cost: " + str(total_cost))