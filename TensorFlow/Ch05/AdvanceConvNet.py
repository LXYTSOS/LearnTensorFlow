# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:39:27 2017

@author: sl169
"""

from cifar import cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3000
batch_size = 128
data_dir = '../cifar10/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

#这里对数据进行了数据增强，随机水平翻转，随机剪切一块24*24大小的图片，
#设置随机亮度和对比度，以及对数据进行标准化
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
        batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)


#样本条数为batch_size,图片尺寸为24*24，颜色通道为RGB三通道
image_holder = tf.placeholder(tf.float32, [batch_size, 24,24,3])
label_holder = tf.placeholder(tf.int32, [batch_size])

#第一个卷积层使用5*5的卷积核大小，3个颜色通道，64个卷积核，标准差为0.05，
#这里不对第一个卷积层进行L2正则化，所以wl设为0
weight1 = variable_with_weight_loss(shape=[5,5,3,64], stddev=5e-2,wl=0.0)
#对输入数据image_holder进行卷积操作，步长均为1
kernel1 = tf.nn.conv2d(image_holder, weight1,[1,1,1,1], padding='SAME')
#把这层的偏置初始化为0，将卷积结果加上偏置，然后时候relu激活函数非线性化
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
#激活函数之后，使用一个尺寸为3*3，步长为2*2的最大池化层
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                       padding='SAME')
#LRN层，模仿了生物神经系统的“侧抑制”机制，对局部神经元的活动创建竞争环境，
#使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
#LRN对ReLU这种没有上边界的激活函数会比较有用，因为它会从附近多个卷积核的响应中挑选
#比较大的反馈，但不适合Sigmoid这种有固定边界并且能抑制过大值的激活函数。
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#创建第二个卷积层，第一个卷积核数量是64（输出64个通道），所以本层卷积核尺寸的第三个
#维度即输入的通道数也需要调整为64，bias初始化0.1，先进行LRN，在最大池化
weight2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2,
                                    wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2,[1,1,1,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1],
                       padding='SAME')

#两个卷积层后，使用一个全连接层，需要把前面两个卷积层输出结果全部flatten，
#使用tf.reshape将每个样本变成一维向量，使用get_shape获取数据扁平化之后的长度，
#使用variable_with_weight_loss对全连接层的weight进行初始化，隐藏节点为384，
#正态分布标准差为0.04，bias初始化为0.1，我们不希望全连接串过拟合，设置一个非零的
#weight loss值0.004
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

#下面这个全连接层和前一层很像，隐藏节点下降一半，其他超参数不变
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192,10], stddev = 1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

#完成了模型interface部分的构建，接下来计算CNN的loss
#把cross_entropy的loss添加到整体losses的collection中，
#最后使用tf.add_n将整体losses的collection中的全部loss求和，得到最终的loss，
#其中包括cross entropy loss和两个全连接层中weight的L2 loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

#将logits节点和label_placeholder传入loss函数获得最终loss
loss = loss(logits, label_holder)
#优化器选择Adam，学习速率设为1e-3
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
#使用in_top_k求输出结果中top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#启动图片数据增强的线程队列，一共使用16个线程进行加速，如果不启动，后续的interface
#和训练都无法开始
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
          feed_dict={image_holder: image_batch, label_holder:label_batch})
    duration = time.time() - start_time
    
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch  = float(duration)
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value,examples_per_sec,sec_per_batch))

num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                           label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)