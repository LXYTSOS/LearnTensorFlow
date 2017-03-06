#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:42:49 2017

@author: liuxiangyu
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist.train.images.shape, mnist.train.labels.shape)
#print(mnist.test.images.shape, mnist.test.labels.shape)
#print(mnist.validation.images.shape, mnist.validation.labels.shape)
#训练集有55000个样本，测试集有10000个样本，验证集有5000个样本
#(55000, 784) (55000, 10)
#(10000, 784) (10000, 10)
#(5000, 784) (5000, 10)

#使用InteractiveSession会将这个session注册成默认的session，之后的运算也默认跑再这个session
#里，不同session之间的数据和运算应该都是相互独立的。
sess = tf.InteractiveSession()
#创建一个Placeholder，即输入数据的地方，第一个参数是数据类型，第二个参数［None，784］代表
#tensor的shape，也就是数据尺寸，None表示步限条数的输入，784代表每条输入是一个784维的向量
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
#损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#优化算法，采用随机梯度下降
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#使用TensorFlow的全局参数初始化器执行run方法。
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))