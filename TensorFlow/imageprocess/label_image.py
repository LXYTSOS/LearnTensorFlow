#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:37:29 2017

@author: liuxiangyu
"""

import matplotlib.pyplot as plt
import tensorflow as tf

#读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("../picture/cat.jpg",'r').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    #处理标注框
    #将图像缩小些，这样可视化能让标注框更加清楚
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    #tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，所以需要先将
    #图像矩阵转化为实数类型。它的输入时一个batch的数据，也就是多张图像组成的思维矩阵
    #所以需要将解码之后的图像矩阵加一维
    batched = tf.expand_dims(
            tf.image.convert_image_dtype(img_data, dtype=tf.float32), 0)
    #给出每一张图像的所有标注框，一个标注框有四个数字分别代表y_min,x_min,y_max,x_max
    #这里给出的数字都是图像的相对位置，比如在180*267的图像中，[0.35,0.47,0.5,0.56]
    #代表了从(63,125)到(90,150)的图像
    boxes = tf.constant([[[0.05,0.05,0.9,0.7], [0.35,0.47,0.5,0.56]]])
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(img_data), bounding_boxes=boxes)
    result = tf.image.draw_bounding_boxes(batched, boxes)
    plt.imshow(result[0].eval())
    plt.show()

    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    plt.imshow(image_with_box[0].eval())
    plt.show()
    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()