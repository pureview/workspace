#import models
import numpy as np
import tensorflow as tf
import cv2


# my
from DataReader import DataReader
from DataReader import DataSets

'''
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
'''
#X_train, Y_train, X_test, Y_test = load_data()

#dim_ordering = K.image_dim_ordering()
K_dim_ordering = 'tf'

IMG_W = 400
IMG_H = 400
IMG_C = 1
batch_size = 32
Y_dim = 4



if K_dim_ordering == 'th':
    input_shape = (IMG_C, IMG_H, IMG_W)
else:
    input_shape = (IMG_H, IMG_W, IMG_C)


#path = '/data2/dataset/eye_conner_0317/eye_conner_0217/eye_conner1_eye_lst-1.txt'
path = '/data2/dataset/eye_conner_0317/eye_conner_0228/txt_list.txt'
path = '/data2/dataset/VR_20170401/VR_20161208_label1/txt_list-label-10w.txt'
path = r"F:/DeepLearning/work201704_tmp/txt_list-label2.txt"

ds = DataSets(path, img_w=IMG_W, img_h=IMG_H, img_c=IMG_C, Y_dim=4)

train_data = ds.TrainData
test_data = ds.TestData
valid_data = ds.ValidData


test_idx = 0


for i in range (0, 1000, batch_size):
    if i + batch_size < 10000:
        test_batch = test_data.get_next_batch(batch_size)

        test_batch_X = test_batch[0]
        test_batch_Y = test_batch[1]
        test_batch_X = test_batch_X.reshape( (batch_size, IMG_H, IMG_W, IMG_C) )
        test_batch_Y = test_batch_Y.reshape( (batch_size, Y_dim) )

        for k in range(batch_size):

            img = test_batch_X[k, :, :, :]
            img = img.reshape(IMG_H, IMG_W, IMG_C)
            img = img*255.0
            #img = img.astype('float32')
            img = img.astype('uint8')
            #y0 = test_batch_Y[k, :]
            y0 = test_batch_Y[k, :]
            
            (ckx0, cky0) = (int(y0[0]*IMG_W), int(y0[1]*IMG_H))
            cv2.circle(img, (ckx0, cky0), 1, (0,255,0), 1)
            
            r = (int(y0[2]*IMG_W/2.0))
            cv2.circle(img, (ckx0, cky0), r, (255,255,255), 1)
            
            cv2.imwrite( './tmp_pics/test-'+str(i)+'-'+str(k)+'.png', img)
            print('write')
            #cv2.imshow('tmp_pics', img)
            #cv2.waitKey(0)
            test_idx = test_idx + 1
            
            if test_idx >= 10:
                break

