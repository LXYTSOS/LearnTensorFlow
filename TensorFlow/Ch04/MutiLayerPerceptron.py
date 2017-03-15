# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:14:14 2017

@author: sl169
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST/MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))


#1、定义算法公式
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

#通过tf.nn.relu实现一个激活函数为ReLU的隐藏层
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
#实现Dropout功能，随机将一部分节点置为0，keep_prob是不值为0的比例
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
#输出层使用Softmax
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

#2、定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.25).minimize(cross_entropy)

#3、开始训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob:0.75})

#4、对模型准确率评测
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))