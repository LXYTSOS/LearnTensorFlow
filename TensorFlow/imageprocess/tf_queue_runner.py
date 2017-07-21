# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:36:24 2017

@author: sl169
"""

import tensorflow as tf

#tf.QueueRunner主要用于启动多个线程来操作同一个队列，启动的这些线程可以通过
#tf.Coordinator类来统一管理

#声明一个先进先出的队列，队列中最多100个元素，类型为实数
queue = tf.FIFOQueue(100, "float")
#定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#使用tf.QueueRunner来创建多个线程运行队列的入队操作
#tf.QueueRunner的第一个参数给出了被操作的队列，[enqueue_op]*5
#表示要启动5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

#将定义过的QueueRunner加入TensorFlow计算图上指定的集合
#tf.train.add_queue_runner函数没有指定集合
#则加入默认集合tf.GraphKeys.Queue_RUNNINGS,
#下面的函数是将刚刚定义的qr加入默认的tf.GraphKeys.Queue_RUNNINGS集合。
tf.train.add_queue_runner(qr)
#定义出队列操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    #使用tf.train.Coordinator来协同启动的线程
    coord = tf.train.Coordinator()
    #使用tf.train.QueueRunner时要明确调用tf.train.start_queue_runners来启动所有
    #线程，否则因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队操作进行
    #tf.train.start_queue_runners函数会默认启动tf.GraphKeys.Queue_RUNNINGS集合中
    #所有的QueueRunner，因为这个函数只支持启动指定集合中的QueueRunner，所以一般来说
    #tf.train.add_queue_runner和tf.train.start_queue_runners函数会指定同一个集合
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #获取队列中的值
    for _ in range(3): print(sess.run(out_tensor)[0])
    
    #使用tf.Coordinator停止所有线程
    coord.request_stop()
    coord.join(threads)

#上面程序将启动5个线程来执行队列入队操作，其中每个线程都是讲随机数写入队列，于是在每次
#运行出队操作时，可以得到一个随机数