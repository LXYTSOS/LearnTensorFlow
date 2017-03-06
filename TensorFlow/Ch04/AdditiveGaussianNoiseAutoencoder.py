#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:40:39 2017

@author: liuxiangyu
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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