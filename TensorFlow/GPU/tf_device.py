# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:19:57 2017

@author: sl169
"""

import tensorflow as tf

#a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
#b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
#c = a + b

# 通过tf.device将运算指定到特定的设备上。 
#with tf.device('/cpu:0'): 
#    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a') 
#    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b') 
#    c = a + b

a_cpu = tf.Variable(0, name="a_cpu")
with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name="a_gpu")
    # 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU上。
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a') 
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b') 
    c = a + b 
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#通过allow_soft_placement参数自动将无法在GPU上的操作放回CPU
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))
