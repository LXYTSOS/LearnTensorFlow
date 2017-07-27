# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:10:47 2017

@author: sl169
"""

from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

#导入TFLearn
learn = tf.contrib.learn

#自定义模型，对于给定的输入数据（features）以及对应的正确答案（target），返回在这些
#数据上的预测值、损失值及训练步骤
def my_model(features, target):
    #将预测的目标转换为one-hot编码形式，因为其中共有三个类别，所以向量长度为3，经过
    #转换后，第一个类别表示为(1,0,0)，第二个为(0,1,0)，第三个为(0,0,1)
    target = tf.one_hot(target, 3, 1, 0)
    
    #定义模型以及其在给定数据上的损失函数，TFLearn通过logistic_regression封装了
    #一个单层全连接神经网络
    logits, loss = learn.models.logistic_regression(features, target)
    
    #创建模型的优化器，并的到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.contrib.framework.get_global_step(), #获取训练步数并在训练时更新
            optimizer='Adagrad',
            learning_rate=0.1)
    
    #返回在给定数据集上的预测结果、损失值以及优化步骤
    return tf.arg_max(logits, 1), loss, train_op

#加载iris数据集，并划分训练集和测试集
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0)

#对自定义模型进行封装
#使用下面这种方式会报Expected sequence or array-like, got <class 'generator'>
#classifier = learn.Estimator(model_fn=my_model)
classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir="Models/model_1"))

#使用封装好的模型和训练数据执行100轮迭代
classifier.fit(x_train, y_train, steps=800)

#使用训练好的模型进行结果预测
#y_predicted = [i for i in classifier.predict(x_test)]
y_predicted = classifier.predict(x_test)

#计算模型的准确度
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.2f%%' % (score * 100))