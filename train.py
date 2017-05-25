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
from yolo_loss import yolo_loss, get_loc, get_iou, get_conf
from models import Model_YOLO20, Model_YOLO


def get_lr(base_lr, batch_num):
    base_lr = float(base_lr)
    if batch_num <= 100:
        return base_lr*10

    if batch_num > 100 and batch_size <= 800:
        return base_lr*1.0

    if batch_size > 800 and batch_num <= 35000:
        return base_lr*0.1

    return base_lr




def get_ret_loc(ret):
    assert ret.ndim==2
    max_idx=[-1,-1]
    max_conf = -1.0
    for i in range(len(ret)):
        for j in range(len(ret[0])):
            if ret[i,j] > max_conf:
                max_conf = ret[i,j]
                max_idx = [i,j]
    return max_idx



GRID_DEPT=6
Y_dim0=4
batch_size=64
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



S = yolo_cfg.get_S()


image_w = tf.constant(IMG_W, name="image_w")
image_h = tf.constant(IMG_H, name="image_h")
image_c = tf.constant(IMG_C, name="image_c")
tensor_config = tf.constant(np.zeros( [100] ), name="tensor_config")


input_image = tf.placeholder('float', shape=[None, IMG_H, IMG_W, IMG_C], name="input_image")
y_ture_label = tf.placeholder('float', shape=[None, S, S, GRID_DEPT], name="y_ture_label")

predictons = Model_YOLO20(input_image)

yolo_loss = yolo_loss(predictons, y_ture_label)
pred_iou = get_iou(predictons, y_ture_label)
learning_rate = tf.placeholder('float', shape=[], name="learning_rate")

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

# Create a summary to monitor cost tensor
tf.summary.scalar("pred_iou", pred_iou)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)


# pred
print("pred:", predictons.name)
tf.summary.histogram(predictons.name, predictons)

print("pred_iou:", pred_iou.name)
tf.summary.histogram(pred_iou.name, pred_iou)




# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

logs_path='/tmp/tfb/'
# op to write logs to Tensorboard
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

avg_cost = 0.0
total_batch = 50000
base_lr = 0.001
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

    train_batch_Y0 = np.zeros([train_batch_Y.shape[0], 2])
    train_batch_Y = np.column_stack( [train_batch_Y, train_batch_Y0] )

    train_batch_Y_feed = np.zeros( [train_batch_Y.shape[0], S, S, GRID_DEPT], dtype=np.float32 )

    for ii in range(train_batch_Y.shape[0]):
        sx,sy,gx,gy = get_loc(train_batch_Y[ii,0], train_batch_Y[ii,1] )
        train_batch_Y_feed[ii, sx, sy, 0:2] = [gx,gy]
        train_batch_Y_feed[ii, sx, sy, 2:4] = np.sqrt(train_batch_Y[ii, 2:4])
        train_batch_Y_feed[ii, sx, sy, 4] = 1.0



    print('feed')
    train_batch_X.astype(np.float32)
    train_batch_Y.astype(np.float32)

    '''
    debug for input data
    for i in range(train_batch_Y.shape[0]):
        img = train_batch_X[i]
        NH=400
        NW=400
        img=img*255.0
        img=img.astype(np.uint8)

        xxx = np.sum(train_batch_Y_feed[i,:,:,4])
        print("sum of conf", xxx)
        (sx,sy) = get_ret_loc(train_batch_Y_feed[i,:,:,4])
        print("loc",(sx,sy))
        print("conf of loc", train_batch_Y_feed[i,sx,sy,4])
        (dx,dy)=train_batch_Y_feed[i,sx,sy,0:2]
        feed_x = (sx+dx)/S
        feed_y = (sy+dy)/S
        feed_w = train_batch_Y_feed[i,sx,sy,2]
        feed_h = feed_w*feed_w
        feed_h = train_batch_Y_feed[i,sx,sy,3]
        feed_h = feed_h*feed_h
        (ix,iy,ir) = (int(feed_x*NW),int(feed_y*NW),int(feed_h*NW/2))
        cv2.circle(img, (ix,iy), ir, (255,0,0))
        cv2.imwrite("./data_chk/"+str(i)+".png", img)

    print("continue!!!!!!!!!!!")
    continue
    '''


    #sess.run(train_op, feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed, learning_rate:0.001} )
    #print("DEBUG!!!!!!")
    #print(train_batch_Y_feed[0,:,:,4])
    #input("pause")
    _, c, train_iou, pred,summary = sess.run([apply_grads, yolo_loss, pred_iou, predictons, merged_summary_op],
            feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed, learning_rate:get_lr(base_lr, i)})
    
    #_, c, summary = sess.run([apply_grads, yolo_loss, merged_summary_op],
    #        feed_dict={ input_image:train_batch_X, y_ture_label:train_batch_Y_feed, learning_rate:get_lr(base_lr, i)})

    
    print("train loss=       ", c)
    print("IOU:              ", train_iou)
    print("cur learning_rate=  ", learning_rate)
    '''
    print("predictions:")
    sam0_pred = pred[0,:,:,4]
    print(np.max(sam0_pred), np.min(sam0_pred), np.sum(sam0_pred))
    print("\ndebug conf")
    print(train_batch_Y_feed[0,:,:,4])
    print(pred[0,:,:,4])
    print("debug conf\n")
    '''

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
        sx,sy,gx,gy = get_loc(test_batch_Y[ii,0], test_batch_Y[ii,1] )
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
