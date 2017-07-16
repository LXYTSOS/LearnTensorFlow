# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:50:52 2017

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
    
    #将数据的类型转化成实数方便样例程序对图像进行处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    
    #通过tf.image.resize_images函数调整图像的大小，第一个参数为原始图像，
    #第二，三个参数为调整后图像的大小，method参数给出了调整图像大小的算法
    #0-双线性插值法（Bilinear interpolation）
    #1-最近邻居法（Nearest neighbor interpolation）
    #2-双三次插值法（Bicubic interpolation）
    #3-面积插值法（Area interpolation）
    resized = tf.image.resize_images(img_data, [300,300], method=3)
    
    #输出调整后图像的大小，此处的结果为(300, 300, ?)表示图像大小是300*300
    #但图像的深度在没有明确设置之前会是问号
    print(resized.get_shape())
#    plt.imshow(resized.eval())
#    plt.show()
    
    #通过tf.image.resize_image_with_crop_or_pad函数调整图像大小，第一个参数是原始图像，
    #后面两个参数是调整后目标图像的大小，如果原始图像尺寸大于目标图像，则自动截取图像居中的部分，
    #如果目标图像大于原始图像，则在原始图像周围填充全0背景
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
#    plt.imshow(croped.eval())
#    plt.show()
#    plt.imshow(padded.eval())
#    plt.show()
    
    #通过比例调整图像大小，tf.image.central_crop第二个参数是调整的比例，取值(0,1]
    central_cropped = tf.image.central_crop(img_data, 0.5)
#    plt.imshow(central_cropped.eval())
#    plt.show()
    
    #图像翻转
    #上下翻转
    flipped_up_down = tf.image.flip_up_down(img_data)
#    plt.imshow(flipped_up_down.eval())
#    plt.show()
    #左右翻转
    flipped_left_right = tf.image.flip_left_right(img_data)
#    plt.imshow(flipped_left_right.eval())
#    plt.show()
    #对角线翻转（类似矩阵转置）
    transposed = tf.image.transpose_image(img_data)
#    plt.imshow(transposed.eval())
#    plt.show()
    #以一定概率翻转图像
    random_flipped_up_down = tf.image.random_flip_up_down(img_data)
    random_flipped_left_right = tf.image.random_flip_left_right(img_data)
    
    #图像色彩调整
    #调整亮度
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    #在[-max_delta, max_delta)的范围随机调整亮度
    max_delta = 0.7
    adjusted = tf.image.random_brightness(img_data, max_delta)
#    plt.imshow(adjusted.eval())
#    plt.show()
    
    #调整对比度
    adjusted = tf.image.adjust_contrast(img_data, -5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    adjusted = tf.image.adjust_contrast(img_data, 5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    #在[lower, upper]范围内随机调整
    adjusted = tf.image.random_contrast(img_data, 1, 10)
#    plt.imshow(adjusted.eval())
#    plt.show()
    
    #调整色相
    adjusted = tf.image.adjust_hue(img_data, 0.1)
#    plt.imshow(adjusted.eval())
#    plt.show()
    adjusted = tf.image.adjust_hue(img_data, 0.3)
#    plt.imshow(adjusted.eval())
#    plt.show()
    adjusted = tf.image.adjust_hue(img_data, 0.6)
#    plt.imshow(adjusted.eval())
#    plt.show()
    adjusted = tf.image.adjust_hue(img_data, 0.9)
#    plt.imshow(adjusted.eval())
#    plt.show()
    #在[-max_delta, max_delta]随机调整,max_delta取值[0,0.5]
    max_delta = 0.3
    adjusted = tf.image.random_hue(img_data, max_delta)
#    plt.imshow(adjusted.eval())
#    plt.show()
    
    #调整饱和度
    adjusted = tf.image.adjust_saturation(img_data, -5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    adjusted = tf.image.adjust_saturation(img_data, 5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    #[lower, upper]范围随机调整
    adjusted = tf.image.random_saturation(img_data, 1, 5)
#    plt.imshow(adjusted.eval())
#    plt.show()
    
    #图像标准化：将代表一张图片的三维矩阵中的数字均值变为0，方差变为1
    adjusted = tf.image.per_image_standardization(img_data)
    plt.imshow(adjusted.eval())
    plt.show()
    