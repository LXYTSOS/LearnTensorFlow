# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:35:04 2017

@author: sl169
"""

import tensorflow as tf

# 获取图片数据
file = open('../RNN/sin.png', 'rb')
data = file.read()
file.close()

# 图片处理
image = tf.image.decode_png(data, channels=4)
image = tf.expand_dims(image, 0)

# 添加到日志中
sess = tf.Session()
writer = tf.summary.FileWriter('../logs/log')
summary_op = tf.summary.image("image1", image)

# 运行并写入日志
summary = sess.run(summary_op)
writer.add_summary(summary)

# 关闭
writer.close()
sess.close()