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
import cv2

# my
from DataReader import DataReader
from DataReader import DataSets
from yolo_cfg import YoloCFG
from models import Model_YOLO20
from yolo_loss import yolo_loss, get_loc
from yolo_pred import get_ret_loc


#tf.device("cpu:6")


def get_lr(base_lr, batch_num):
    base_lr = float(base_lr)
    if batch_num <= 100:
        return base_lr*10

    if batch_num > 100 and batch_size <= 25000:
        return base_lr*1.0

    if batch_size > 25000 and batch_num <= 35000:
        return base_lr*0.1

    return base_lr




GRID_DEPT=6
path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)

#path = '/data2/dataset/eye_conner_0317/eye_conner_0228/txt_list.txt'
path = '/data2/dataset/VR_20170401/VR_20161208_label1/txt_list-label-10w.txt'
#path = r'F:\DeepLearning\work201704_tmp\txt_list-label2.txt'
#path='/home/nfm/test/prun/tf_pupil/txt_list1.txt'

IMG_W = yolo_cfg.get_IMG_W()
IMG_H = yolo_cfg.get_IMG_H()
IMG_C = yolo_cfg.get_IMG_C()

ds = DataSets(path, img_w=IMG_W, img_h=IMG_H, img_c=IMG_C, Y_dim=4)

train_data = ds.TrainData
test_data = ds.TestData
valid_data = ds.ValidData


Y_dim0=4
batch_size=32
S = yolo_cfg.get_S()


image_w = tf.constant(IMG_W, name="image_w")
image_h = tf.constant(IMG_H, name="image_h")
image_c = tf.constant(IMG_C, name="image_c")
tensor_config = tf.constant(np.zeros( [100] ), name="tensor_config")


input_image = tf.placeholder('float', shape=[None, IMG_H, IMG_W, IMG_C], name="input_image")
y_ture_label = tf.placeholder('float', shape=[None, S, S, GRID_DEPT], name="y_ture_label")

predictons = Model_YOLO20(input_image)

yolo_loss = yolo_loss(predictons, y_ture_label)
learning_rate = tf.placeholder('float', shape=[], name="learning_rate")


#train_op = tf.train.AdamOptimizer(1e-4).minimize( yolo_loss )

# Gradient Descent  
optimizer = tf.train.GradientDescentOptimizer(learning_rate)  
# Op to calculate every variable gradient  
grads = tf.gradients(yolo_loss, tf.trainable_variables())  
grads = list(zip(grads, tf.trainable_variables()))  
# Op to update all variables according to their gradient  
apply_grads = optimizer.apply_gradients(grads_and_vars=grads)  

#train_op = tf.train.AdamOptimizer(1e-4).minimize( yolo_loss )

sess = tf.Session()
init_op = tf.global_variables_initializer()

sess.run(init_op)

print("Start training!")



saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint("./ckpt")
train_idx = 0
if checkpoint:
    print("Restoring from checkpoint", checkpoint)
    saver.restore(sess, checkpoint)
    train_idx = int(checkpoint.split('-')[-1])
else:
    print("Couldn't find checkpoint to restore from. Starting over.")





# Create a summary to monitor cost tensor
tf.summary.scalar("yolo_loss", yolo_loss)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

logs_path='/tmp/tfb/'
# op to write logs to Tensorboard
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())




print("TEST!!!!!!")
for i in range(400):
    train_batch = train_data.get_next_batch(batch_size)
    train_batch_X = train_batch[0]
    train_batch_Y = train_batch[1]
    train_batch_X.astype(np.float32)
    train_batch_Y.astype(np.float32)

    print(train_batch_X.shape)
    train_batch_X = train_batch_X.reshape( (batch_size, IMG_H, IMG_W, IMG_C) )
    train_batch_Y = train_batch_Y.reshape( (batch_size, Y_dim0) )
    train_batch_Y00 = train_batch_Y.reshape( (batch_size, Y_dim0) )

    train_batch_Y0 = np.zeros([train_batch_Y.shape[0], 2])
    train_batch_Y = np.column_stack( [train_batch_Y, train_batch_Y0] )

    train_batch_Y_feed = np.zeros( [train_batch_Y.shape[0], S, S, GRID_DEPT], dtype=np.float32 )
    for ii in range(train_batch_Y.shape[0]):
        sx,sy,gx,gy = get_loc(train_batch_Y00[ii,0], train_batch_Y00[ii,1] )
        train_batch_Y_feed[ii, sx, sy, :] = train_batch_Y[ii, :]
        train_batch_Y_feed[ii, sx, sy, 0:2] = [gx,gy]

    pred = sess.run(predictons, feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed, learning_rate:1.0} )

    print("pred conf")
    print(pred[0,:,:,4])
    ret_loc = get_ret_loc(pred)
    print("ret_loc")
    print(ret_loc)
    ret_loc0 = ret_loc[0]
    print("get loc0:", ret_loc0)
    sx,sy,gx,gy = get_loc(train_batch_Y00[0, 0], train_batch_Y00[0, 1] )
    print("feed:", train_batch_Y00[0])
    
    [pred_x,pred_y,pred_w,pred_h] = pred[0, ret_loc0[0], ret_loc0[1], 0:4]
    pred_x = (pred_x+ret_loc0[0])/S
    pred_y = (pred_y+ret_loc0[1])/S
    print("pred:", [pred_x,pred_y,pred_w,pred_h])
    print("pred:", pred[0, ret_loc0[0], ret_loc0[1], :])
    
    
    img = train_batch_X[0]
    #print(img.shape)
    NH=400
    NW=400
    img=img*255.0
    img=img.astype(np.uint8)
    pred_w = pred_w*pred_w
    pred_h = pred_h * pred_h
    (ix,iy,ir) = (int(pred_x*NW),int(pred_y*NW),int(pred_w*NW/2))
    cv2.circle(img, (ix,iy), ir, (255,0,0))
    cv2.imwrite("./res_pic/"+str(i)+".png", img)

    #input("next!")




input("return OK")



avg_cost = 0.0
total_batch = 10000
i=train_idx
while(True):

    if i > total_batch:
        break

    i = i + 1

    train_batch = train_data.get_next_batch(batch_size)

    train_batch_X = train_batch[0]
    train_batch_Y = train_batch[1]
    train_batch_X.astype(np.float32)
    train_batch_Y.astype(np.float32)

    print(train_batch_X.shape)
    train_batch_X = train_batch_X.reshape( (batch_size, IMG_H, IMG_W, IMG_C) )
    train_batch_Y = train_batch_Y.reshape( (batch_size, Y_dim0) )

    train_batch_Y0 = np.ones([train_batch_Y.shape[0], 2])
    train_batch_Y = np.column_stack( [train_batch_Y, train_batch_Y0] )

    print(train_batch_Y.shape)


    train_batch_Y_feed = np.zeros( [train_batch_Y.shape[0], S, S, GRID_DEPT], dtype=np.float32 )

    for ii in range(train_batch_Y.shape[0]):
        sx,sy,gx,gy = get_loc(train_batch_Y[ii,0,], train_batch_Y[ii,1] )
        train_batch_Y_feed[ii, sx, sy, :] = train_batch_Y[ii, :]
        train_batch_Y_feed[ii, sx, sy, 0:2] = [gx,gy]


    print('feed')
    train_batch_X.astype(np.float32)
    train_batch_Y.astype(np.float32)

    #sess.run(train_op, feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed, learning_rate:0.001} )

    _, c, summary = sess.run([apply_grads, yolo_loss, merged_summary_op],
            feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed, learning_rate:0.001})

    print("train loss=       ", c)

    # Write logs at every iteration
    summary_writer.add_summary(summary, i)
    # Compute average loss
    avg_cost += c / total_batch

    '''
    test_batch = test_data.get_next_batch(batch_size)
    test_batch_X = test_batch[0]
    test_batch_Y = test_batch[1]
    test_batch_X = test_batch_X.reshape( (batch_size, IMG_H, IMG_W, IMG_C) )
    test_batch_Y = test_batch_Y.reshape( (batch_size, Y_dim0) )

    test_batch_Y0 = np.ones([test_batch_Y.shape[0], 2])
    test_batch_Y = np.column_stack( [test_batch_Y, test_batch_Y0] )

    test_batch_Y_feed = np.zeros( [test_batch_Y.shape[0], S, S, GRID_DEPT], dtype=np.float32 )
    for ii in range(test_batch_Y.shape[0]):
        sx,sy,gx,gy = get_loc(test_batch_Y[ii,0,], test_batch_Y[ii,1] )
        test_batch_Y_feed[ii, sx, sy, :] = test_batch_Y[ii, :]
        test_batch_Y_feed[ii, sx, sy, 0:2] = [gx,gy]

    print(train_batch_X.dtype)
    #test_loss = train_op.run(feed_dict={ input_image:test_batch_X, y_ture_label:test_batch_Y_feed} )
    tt1=np.zeros([batch_size, IMG_H, IMG_W, IMG_C], dtype=np.float32)
    test_loss = sess.run(yolo_loss, feed_dict={ input_image:tt1, y_ture_label:test_batch_Y_feed, learning_rate:0.001} )
    print("i=",i,test_loss)

    print('\n\n')
    '''

    saver.save(sess, './ckpt/progress', global_step=i)


    '''
    if i % 10 ==0:
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output_xy'])
        with tf.gfile.FastGFile('./pb/pupil'+str(i)+'.pb', mode = 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    '''

'''
output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output_xy'])
with tf.gfile.FastGFile('./pb/pupil_last.pb', mode = 'wb') as f:
    f.write(output_graph_def.SerializeToString())

'''
