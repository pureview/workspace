# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:34:53 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-

from __future__ import print_function
import os,sys,math
import numpy as np
import tensorflow as tf

# my
from DataReader import DataReader
from DataReader import DataSets
from yolo_cfg import YoloCFG
from yolo_loss import yolo_loss, get_loc



def weight_varialbe(shape, name):
    init_val = tf.truncated_normal( shape, stddev=0.1, name=name )
    return tf.Variable(init_val)


def bias_variable(shape, name):
    init_val = tf.constant( 0.1, shape=shape, name=name )
    return tf.Variable(init_val)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool(x, ksize, stride):
    return tf.nn.max_pool(x, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding='SAME')


def fc_layer(inputs,hiddens,flat = False,linear = False, name="fc"):
    input_shape = inputs.get_shape().as_list()        
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_transposed = tf.transpose(inputs,(0,3,1,2))
        inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
    else:
        dim = input_shape[1]
        inputs_processed = inputs
    weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1),name=name+"_weight")
    biases = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=name+"_bais")    
    if linear : return tf.add(tf.matmul(inputs_processed,weight),biases)
    #if linear : return tf.nn.sigmoid(tf.add(tf.matmul(inputs_processed,weight),biases) ,name=name)
    ip = tf.add(tf.matmul(inputs_processed,weight),biases)
    alpha=0.1
    return tf.maximum(alpha*ip,ip,name=name)


GRID_DEPT=6
path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)


IMG_W = yolo_cfg.get_IMG_W()
IMG_H = yolo_cfg.get_IMG_H()
IMG_C = yolo_cfg.get_IMG_C()


def Model_YOLO(x):

    with tf.name_scope("model"):
        #cov1
        W_conv1 = weight_varialbe( [3,3,yolo_cfg.get_IMG_C(),4], name="conv1_w")
        b_conv1 = bias_variable( [4] , name="bias1_w")
        h_conv1=tf.nn.relu6( conv2d(x, W_conv1) + b_conv1 )

        #cov2
        W_conv2 = weight_varialbe( [3,3,4,8], name="conv2_w")
        b_conv2 = bias_variable( [8] , name="bias2_w")
        h_conv2=tf.nn.relu6( conv2d(h_conv1, W_conv2) + b_conv2 )
        #pool1
        h_pool1 = max_pool(h_conv2, 8, 8)
        #cov3
        W_conv3 = weight_varialbe( [3,3,8,8], name="conv3_w")
        b_conv3 = bias_variable( [8] , name="bias3_w")
        h_conv3=tf.nn.relu6( conv2d(h_pool1, W_conv3) + b_conv3 )
        #pool1
        h_pool2 = max_pool(h_conv3, 4, 4)
        
        #cov4
        W_conv4 = weight_varialbe( [3,3,8,6], name="conv4_w")
        b_conv4 = bias_variable( [6] , name="bias4_w")
        h_conv4=tf.nn.relu6( conv2d(h_pool2, W_conv4) + b_conv4 )
    return h_conv4


def Model_YOLO20(x):

    with tf.name_scope("model"):
        #cov1
        W_conv1 = weight_varialbe( [3,3,yolo_cfg.get_IMG_C(),4], name="conv1_w")
        b_conv1 = bias_variable( [4] , name="bias1_w")
        h_conv1=tf.nn.relu6( conv2d(x, W_conv1) + b_conv1 )
        
        #h_conv1=tf.contrib.keras.layers.LeakyReLU( conv2d(x, W_conv1) + b_conv1  )
        #cov2
        W_conv2 = weight_varialbe( [3,3,4,8], name="conv2_w")
        b_conv2 = bias_variable( [8] , name="bias2_w")
        h_conv2=tf.nn.relu6( conv2d(h_conv1, W_conv2) + b_conv2 )
        #cov3
        W_conv3 = weight_varialbe( [3,3,8,8], name="conv3_w")
        b_conv3 = bias_variable( [8] , name="bias3_w")
        h_conv3=tf.nn.relu6( conv2d(h_conv2, W_conv3) + b_conv3 )
        #pool1
        h_pool1 = max_pool(h_conv3, 2, 2)
        #cov4
        W_conv4 = weight_varialbe( [3,3,8,16], name="conv4_w")
        b_conv4 = bias_variable( [16] , name="bias4_w")
        h_conv4=tf.nn.relu6( conv2d(h_pool1, W_conv4) + b_conv4 )
        #pool2
        h_pool2 = max_pool(h_conv4, 2, 2)
        #cov5
        W_conv5 = weight_varialbe( [3,3,16,32], name="conv5_w")
        b_conv5 = bias_variable( [32] , name="bias5_w")
        h_conv5=tf.nn.relu6( conv2d(h_pool2, W_conv5) + b_conv5 )
        #pool3
        h_pool3 = max_pool(h_conv5, 2, 2)
        #cov6
        W_conv6 = weight_varialbe( [3,3,32,64], name="conv6_w")
        b_conv6 = bias_variable( [64] , name="bias6_w")
        h_conv6=tf.nn.relu6( conv2d(h_pool3, W_conv6) + b_conv6 )
        #pool4
        h_pool4 = max_pool(h_conv6, 2, 2)
        #cov7
        W_conv7 = weight_varialbe( [3,3,64,128], name="conv7_w")
        b_conv7 = bias_variable( [128] , name="bias7_w")
        h_conv7=tf.nn.relu6( conv2d(h_pool4, W_conv7) + b_conv7 )
        
        #cov8
        W_conv8 = weight_varialbe( [3,3,128,64], name="conv8_w")
        b_conv8 = bias_variable( [64] , name="bias8_w")
        h_conv8=tf.nn.relu6( conv2d(h_conv7, W_conv8) + b_conv8 )
        #cov9
        W_conv9 = weight_varialbe( [3,3,64,32], name="conv9_w")
        b_conv9 = bias_variable( [32] , name="bias9_w")
        h_conv9=tf.nn.relu6( conv2d(h_conv8, W_conv9) + b_conv9 )
        #pool
        h_pool5 = max_pool(h_conv9, 2, 2)
        #cov10
        W_conv10 = weight_varialbe( [3,3,32,16], name="conv10_w")
        b_conv10 = bias_variable( [16] , name="bias10_w")
        h_conv10=tf.nn.relu6( conv2d(h_pool5, W_conv10) + b_conv10 )
        #cov11
        W_conv11 = weight_varialbe( [3,3,16,16], name="conv11_w")
        b_conv11 = bias_variable( [16] , name="bias11_w")
        h_conv11=tf.nn.relu6( conv2d(h_conv10, W_conv11) + b_conv11 )
        
        #cov12
        W_conv12 = weight_varialbe( [1,1,16,6], name="conv12_w")
        b_conv12 = bias_variable( [6] , name="bias12_w")
        h_conv12=tf.nn.sigmoid( conv2d(h_conv11, W_conv12) + b_conv12 )
        #h_conv12 = conv2d(h_conv11, W_conv12) + b_conv12
        
        h_conv120 = tf.reshape(h_conv12, [-1, 13*13*6])
        weight = tf.Variable(tf.truncated_normal([13*13*6, 13*13*6], stddev=0.1),name="FC"+"_weight")
        biases = tf.Variable(tf.constant(0.1, shape=[13*13*6]),name="FC"+"_bais")
        linear = tf.add(tf.matmul(h_conv120,weight),biases)
        linear = tf.reshape(linear, [-1,13,13,6])

        #fc1 = fc_layer(h_conv7, 256, flat=True, linear = False, name="fc1")
        #fc11 = fc_layer(fc1, 246, flat=False, linear = False, name="fc11")
        #fc2 = fc_layer(fc11, 13*13*6, flat=False, linear = True, name="fc2")
        #fc22=tf.reshape(fc2, [-1,13,13,6])
        #fc22=tf.clip_by_value(fc22, 0.001, 0.999)
    return h_conv12

