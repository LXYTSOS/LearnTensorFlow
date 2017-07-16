# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:29:02 2017

@author: sl169
"""

import matplotlib.pyplot as plt
import tensorflow as tf

#读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("../picture/cat.jpg",'r').read()

with tf.Session() as sess:
    #将图像使用jpeg的格式解码从而得到图像对应的三维矩阵，TensorFlow还提供了
    #tf.image.decode_png函数对png格式的图像进行解码，解码之后的结果为一个张量
    #在使用它的取值之前要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)
    
    #输出解码之后的三维矩阵
    print(img_data.eval())
    
    #使用pyplot工具可视化得到图像
    plt.imshow(img_data.eval())
    plt.show()
    
    #将数据的类型转化成实数方便样例程序对图像进行处理
    #保存jpg文件时dtype设定成uint8,用于后面调整大小时设定成float32
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
    
    #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("../picture/output.jpg", "wb") as f:
        f.write(encoded_image.eval())