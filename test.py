# -*- coding: utf-8 -*-

from __future__ import print_function
import os,sys,math
import numpy as np
import tensorflow as tf

# my
from DataReader import DataReader
from DataReader import DataSets
from yolo_cfg import YoloCFG
from model_cfg import ModelCFG
#from model import Model_YOLO
from yolo_loss import yolo_loss, get_loc
from yolo_pred import yolo_predictions
from models import Model_YOLO20


GRID_DEPT=6
path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)

tf.device("/cpu:1")


path = '/data2/dataset/eye_conner_0317/eye_conner_0228/txt_list.txt'
path = '/data2/dataset/VR_20170401/VR_20161208_label1/txt_list-label-10w.txt'
#path = r'F:\DeepLearning\work201704_tmp\txt_list-label2.txt'

IMG_W = yolo_cfg.get_IMG_W()
IMG_H = yolo_cfg.get_IMG_H()
IMG_C = yolo_cfg.get_IMG_C()

ds = DataSets(path, img_w=IMG_W, img_h=IMG_H, img_c=IMG_C, Y_dim=4)

train_data = ds.TrainData
test_data = ds.TestData
valid_data = ds.ValidData


Y_dim0=4
batch_size=1


S = yolo_cfg.get_S()


image_w = tf.constant(IMG_W, name="image_w")
image_h = tf.constant(IMG_H, name="image_h")
image_c = tf.constant(IMG_C, name="image_c")
tensor_config = tf.constant(np.zeros( [100] ), name="tensor_config")


input_image = tf.placeholder('float', shape=[None, IMG_H, IMG_W, IMG_C], name="input_image")
y_ture_label = tf.placeholder('float', shape=[None, S, S, GRID_DEPT], name="y_ture_label")
predictons = Model_YOLO20(input_image)

with tf.name_scope("main_yolo_loss"):
    yolo_loss = yolo_loss(predictons, y_ture_label)
    print("main_yolo_loss")
    print(yolo_loss)
    tf.Print(yolo_loss, [yolo_loss])


sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
print("Start test!")

saver = tf.train.Saver()

checkpoint = tf.train.latest_checkpoint("./ckpt")

if checkpoint:
    print("Restoring from checkpoint", checkpoint)
    saver.restore(sess, checkpoint)
else:
    print("Couldn't find checkpoint to restore from. Starting over.")


def get_ret_loc(ret):
    conf = ret[:,:,:,4]
    final=[]
    for i in range(len(conf)):
        max_conf=0
        max_index=[]
        for j in range(len(conf[i])):
            for k in range(len(conf[i][j])):
                if conf[i][j][k]>max_conf:
                    max_conf=conf[i][j][k]
                    max_index=[i,j,k]
        final.append(max_index)
    return np.array(final)




for i in range(4000):
    train_batch = train_data.get_next_batch(batch_size)
    train_batch_X = train_batch[0]
    train_batch_Y = train_batch[1]
    train_batch_X.astype(np.float32)
    train_batch_Y.astype(np.float32)

    print(train_batch_X.shape)
    print(train_batch_Y.shape)
    train_batch_X = train_batch_X.reshape( (batch_size, IMG_H, IMG_W, IMG_C) )
    rain_batch_Y = train_batch_Y.reshape( (batch_size, Y_dim0) )
    train_batch_Y_feed = np.zeros( [train_batch_Y.shape[0], S, S, GRID_DEPT], dtype=np.float32 )
    pred = sess.run(predictons, feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed} )

    print(pred.shape)
    print(pred[:,:,:,4])

    #ret111 = get_ret(pred)
    #print('ret111.shape',ret111.shape)
    #print('ret111[0]:',ret111[0])

    ret_loc = get_ret_loc(pred)
    print("get loc:", ret_loc)
    feed_loc = get_ret_loc(pred)
    print(train_batch_Y)
    sx,sy,gx,gy = get_loc(train_batch_Y[0,0], train_batch_Y[0,1] )
    print("feed:", [sx, sy])
    input("pause")
