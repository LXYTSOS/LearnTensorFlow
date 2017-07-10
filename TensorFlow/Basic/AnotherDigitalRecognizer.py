# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:36:55 2017

@author: sl169
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络的参数
#一个隐藏层，500个节点
LAYER1_NODE = 500
BATCH_SIZE = 100

#基础学习率
LEARNING_RATE_BASE = 0.8
#学习率的衰减率
LEARNING_RATE_DECAY = 0.99
#描述模型复杂度的正则化项在损失函数中的系数。
REGULARIZATION_RATE = 0.0001
#训练轮数
TRAINING_STEPS = 30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

#给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        #计算隐藏层向前传播的结果，使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        
        #计算输出层向前传播结果，因为在计算损失函数时会一并计算softmax函数，
        #所以这里不需要加入激活函数
        return tf.matmul(layer1, weights2) + biases2
    else:
        #首先使用avg_class.average函数来计算得出变量的平均滑动值，
        #然后再计算相应的神经网络前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
        + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    
    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    #计算在当前参数下神经网络向前传播的结果，这里给出的用于计算滑动平均的类为None，
    #所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    
    global_step = tf.Variable(0, trainable=False)
    
    #给定滑动平均衰减率和训练轮数，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, 
                                                          global_step)
    
    #tf.trainable_variables()返回的是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素。
    #这个集合的元素是所有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #计算使用了滑动平均之后的前向传播结果。
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    #计算交叉熵作为刻画预测值与真实值之间差距的损失函数,
    #使用sparse_softmax_cross_entropy_with_logits来计算交叉熵，
    #当分类问题只有一个正确答案时，会加速交叉熵的计算
    #tf.argmax得到数组最大值的下标
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    
    #计算在当前batch中所有样例交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失加正则化损失
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                  global_step=global_step)
    
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    #校验使用了滑动平均模型的神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    #将布尔型数值转换成实数型，然后计算平均值，也就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #准备验证数据，用来大致判断停止的条件和评判训练结果
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        
        #准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is %g "
                      "test accuracy using average model is %g "%
                      (i, validate_acc, test_acc))
            
            #产生这一batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        
        #训练结束后，在测试数据上检测神经网络模型的最终正确率
#        test_acc = sess.run(accuracy, feed_dict=test_feed)
#        print("After %d training step(s), test accuracy using average model is %g " %
#              (TRAINING_STEPS, test_acc))
        #保存训练模型
        saver = tf.train.Saver()
        saver.save(sess, "../model/anotherDigitalRecognizer.ckpt")

#主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST/MNIST_data/", one_hot=True)
    train(mnist)

#TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()