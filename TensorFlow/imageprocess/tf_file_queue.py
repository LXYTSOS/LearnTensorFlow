# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:40:10 2017

@author: sl169
"""

import tensorflow as tf

#使用tf.train.match_filnames_once函数获取文件列表
files = tf.train.match_filenames_once("../tfrecordfiles/data.tfrecords-*")

#通过tf.train.string_input_producer函数创建输入队列，输入队列中的文件列表为
#tf.train.match_filenames_once函数获取的文件列表，这里将shuffle设置为False
#来避免随机打乱读文件顺序，但一般在解决实际问题时，会将shuffle设置为True
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
        serialized_example,
        features={
                'i': tf.FixedLenFeature([], tf.int64),
                'j': tf.FixedLenFeature([], tf.int64),
        })

example, label = features['i'], features['j']
#一个batch中样例个数
batch_size = 2
#组合样例队列中最多可以存储的样例数，如果队列太大，需要占用很多内存资源，如果太小
#那么出队操作可能会因为没有数据而被阻塞，从而导致训练效率降低，一般来说这个队列大小会和
#每一个batch的大小有关
capacity = 1000 + 3 * batch_size

#使用tf.train.batch来组合样例，[example, label]给出了需要组合的元素
#batch_size给出了每个batch中样例的个数，capacity给出了队列的最大容量，当队列长度等于
#最大容量时，TensorFlow将暂停入队操作，而只是等待元素出队，当队列长度小于最大容量时，
#TensorFlow将自动重新启动入队操作
example_batch, label_batch = tf.train.batch(
        [example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    #虽然在这里没有声明任何变量，但使用tf.train.match_filenames_once函数时需要
    #初始化一些变量
#    tf.global_variables_initializer().run()
#    print(sess.run(files))
#    
#    #声明tf.Coordinator类来协同不同线程，并启动线程
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#    
#    #多次执行获取数据的操作
#    for i in range(6):
#        print(sess.run([features['i'], features['j']]))
#    coord.request_stop()
#    coord.join(threads)
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)