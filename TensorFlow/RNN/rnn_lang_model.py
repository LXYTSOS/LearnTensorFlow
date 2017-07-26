# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:12:13 2017

@author: sl169
"""

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

#数据存放的路径
DATA_PATH = "../data/PTB/"
#隐藏层大小
HIDDEN_SIZE = 200
#深层循环神经网络中LSTM结构的层数
NUM_LAYERS = 2
#词典规模，加上语句结束标识符和稀有单词标识符
VOCAB_SIZE = 10000

#学习速率
LEARNING_RATE = 1.0
#训练数据batch大小
TRAIN_BATCH_SIZE = 20
#训练数据截断长度
TRAIN_NUM_STEP = 35

#在测试时不需要使用截断，所以可以将测试数据看成一个超长的序列
#测试数据batch大小
EVAL_BATCH_SIZE = 1
#测试数据截断长度
EVAL_NUM_STEP = 1
#使用训练数据的轮数
NUM_EPOCH = 2
#节点不被dropout的概率
KEEP_PROB = 0.5
#用于控制梯度膨胀的参数
MAX_GRAD_NORM = 5

#通过一个PTBModel类来描述模型，这样方便维护循环神经网络中的状态
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        #记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        #定义输入层，可以看到输入层的维度为batch_size * num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        #定义预期输出
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        #定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell,output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS)
        
        #初始化最初的状态，也就是全0的向量
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        #将单词ID转换成单词向量，因为一共有VOCAB_SIZE个单词，每个单词的维度为
        #HIDDEN_SIZE，所以embedding参数的维度为VOCAB_SIZE*HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        
        #将原本batch_size*num_steps个单词ID转化成单词向量，转化后的输入层维度为
        #batch_size*num_steps*HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        #只是在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)
        
        #定义输出列表，在这里先将不同时刻LSTM结构的输出收集起来，再通过一个全连接层
        #得到最终输出
        outputs = []
        #state存储不同batch中LSTM的状态，将其初始化为0
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                #从输入数中获取当前时刻的输入并传入LSTM结构
                cell_output, state = cell(inputs[:, time_step, :], state)
                #将当前输出加入输出队列
                outputs.append(cell_output)
        
        #把输出队列展开成[batch, HIDDEN_SIZE*num_steps]的形状，然后再
        #reshape成[batch*num_steps, HIDDEN_SIZE]的形状
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        
        #将从LSTM中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在
        #每一个时刻上都是一个长度为VOCAB_SIZE的数组，经过softmax层后表示下一个
        #位置是不同单词的概率
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        
        #定义交叉熵损失函数，TensorFlow提供了sequence_loss_by_example来计算一个
        #序列的交叉熵和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],                           #预测的结果
                [tf.reshape(self.targets, [-1])],   #期待的正确答案，这里将
                                                    #[batch_size, num_steps]
                                                    #二维数组压缩成一维
                #损失的权重，这里所有的权重都为1
                [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        
        #计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        
        #只在训练时定义反向传播操作
        if not is_training: return
        
        trainable_variables = tf.trainable_variables()
        #通过clip_by_global_norm函数控制梯度大小，避免梯度膨胀
        grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        
        #定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        
        #定义训练步骤
        self.train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables))

#使用给定的模型model，在数据data上运行train_op并返回在全部数据上的perplexity
def run_epoch(session, model, data, train_op, output_log):
    #计算perpliexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    #使用当前数据训练或者测试模型
    for step, (x, y) in enumerate(
            reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        #在当前batch上运行train_op并计算损失，交叉熵损失函数计算的就是下一个单词
        #为给定单词的概率
        cost, state, _ = session.run(
                [model.cost, model.final_state, train_op],
                {model.input_data: x, model.targets: y,
                 model.initial_state: state})
        #将不同时刻，不同batch的概率加起来就可以得到第二个perplexity边的部分
        total_costs += cost
        iters += model.num_steps
        
        #只有在训练时输出日志
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                    step, np.exp(total_costs / iters)))
    #返回给定模型在给定数据上的perplexity
    return np.exp(total_costs / iters)

def main(_):
    #获取原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    
    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    
    #定义训练用的循环神经网络模型
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    
    #定义评测用的循环神经网络模型
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        
        #使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            #在所有训练数据上训练循环神经网络模型
            run_epoch(session, train_model, train_data, train_model.train_op, True)
            
            #使用验证数据评测模型效果
            valid_perplexity = run_epoch(
                    session, eval_model, valid_data, tf.no_op(), False)
            print("Epoch: %d Validation Perplexity: %.3f" % (
                    i + 1, valid_perplexity))
        
        #最后使用测试数据测试模型效果
        test_perplexity = run_epoch(
                session, eval_model, test_data, tf.no_op(), False)
        print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    tf.app.run()