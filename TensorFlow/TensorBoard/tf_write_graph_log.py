# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:23:41 2017

@author: sl169
"""

import tensorflow as tf
input1 = tf.constant([1.0, 2.0, 3.0], name="input2")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")
#with tf.name_scope("input1"):
#    input1 = tf.constant([1.0, 2.0, 3.0], name="input2")
#with tf.name_scope("input2"):
#    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
#output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("../logs/graphlog", tf.get_default_graph())
writer.close()
