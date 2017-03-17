# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:37:40 2017

@author: sl169
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST/MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#ksize:池化窗口大小，一般是[1,height,width,1]，不想对batch和channels上做池化，
#所以这两个维度设为1.
#strides：窗口在每一个维度上滑动步长，一般也是[1,strides,strides,1]
#tf.nn.conv2d是TensorFlow中的2维卷积函数，x是输入，W是卷积参数
#比如[5,5,1,32]前两个表示卷积核的尺寸，第三个表示通道个数，第四个表示卷积核个数
#strides表示卷积模板移动的步长，都是1表示不会遗漏的划过图片的每一个点
#padding表示你边界处理的方式，SAME表示给边界加上padding让卷积的输出输入保持同样的尺寸
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#tf.nn.max_pool是TensorFlow中的最大池化函数，使用2×2最大池化，就是将一个2×2的像素块
#降为1×1的像素
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
#将1维输入向量转为2维，即从784转为28*28，-1表示样本数量不固定，1表示颜色通道数量
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#现在定义第二个卷积层，卷积核数量为64，也就是说这一层卷积会提取64种特征。
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#前面经历了两次步长为2*2的最大池化，所以边长已经只有1/4了，图片尺寸由28*28变成7*7
#第二个卷积层的卷积核数量是64，输出的tensor尺寸为7*7*64，将其从1维变形成2维，
#隐藏节点1024，使用ReLU激活函数
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为减轻过拟合，使用一个dropout层，在训练时随机丢弃一部分节点，预测时保留全部数据来追求
#最好的预测性能
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#最后将dropout层的输出连接一个softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义损失函数和优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

#定义评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1],
                                                  keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

#全部训练完了后，在最终的测试集上进行全面测试，得到整体的分类准确率
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,
                                                  y_:mnist.test.labels,
                                                  keep_prob:1.0}))