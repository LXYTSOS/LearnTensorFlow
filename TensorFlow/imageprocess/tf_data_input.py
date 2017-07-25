# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:55:47 2017

@author: sl169
"""

import tensorflow as tf

#创建文件列表，并通过文件列表创建输入文件队列，在调用数据处理流程前，需要统一所有
#原始数据的格式，并将它们存储到TFRecord文件中，下面给出的文件列表应该包含所有提供
#训练数据的TFRecord文件
files = tf.train.match_filenames_once("../tfrecordfiles/file_pattern-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

#假设image存储的是图像原始数据，label为样例所对应的标签，height，width，channels给出
#了图片的维度。
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })
decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'],tf.int32)
#pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])

#将处理后的图像和标签数据通过tf.train.shuffle_batch整理成神经网络训练时需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
        [images, labels], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)

#定义神经网络的结构以及优化过程，image_batch可以作为输入提供给神经网络的输入层，
#label_batch则提供了输入batch中样例的正确答案
def inference(input_tensor, weights1, biases1, weights2, biases2):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)
    
# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
                  
        sess.run(train_step) 
    coord.request_stop()
    coord.join(threads)
