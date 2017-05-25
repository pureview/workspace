# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:03:18 2017

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import cv2
import random
#import h5py
import time
import json
import numpy as np
import tensorflow as tf


class DataReader(object):
    def __init__(self, lst, img_w = 227, img_h = 227, img_c = 3, Y_dim=1):
        if len(lst) == 0:
            print("lst length is 0" )
            return None
        self.lst = lst
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c
        self.Y_dim = Y_dim
        self.pic_idx = 0


    def get_sample_number(self):
        return len(self.lst)


    def read_classes_names(self, lst):
        if lst == [] or lst == None:
            return None
        self.classes_lst = lst


    def get_label(self, pic_idx):
        line = self.lst[pic_idx]
        v = line.split('.bmp')
        if (len(v) != 2):
            print("labes txt is wrong!")
            return None

        vlabel = v[1].strip().split(' ')
        y0 = np.zeros(self.Y_dim)
        if self.Y_dim != len(vlabel):
            print("label lenth != Y_dim")
            return None

        for j in range(len(vlabel)):
            y0[j] = float(vlabel[j])

        y0 = y0[np.newaxis,]
        return y0


    def get_next_batch(self, batch_size):
        if self.pic_idx + batch_size > self.get_sample_number():
            self.pic_idx = 0

        X = np.zeros( (0, self.img_w*self.img_h*self.img_c), dtype=np.float32 )
        Y = np.zeros( (0, self.Y_dim), dtype=np.float32 )
        for i in range(batch_size):
            line = self.lst[self.pic_idx]
            v = line.split('.bmp')
            img_name = v[0]+'.bmp'
            #print(img_name)
            if not os.path.exists(img_name):
                print("%s not exists" % img_name)
                return None
            if self.img_c == 1:
                img0 = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            else:
                img0 = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img0, (self.img_h, self.img_w), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            #img = img.reshape( (-1, self.img_w, self.img_h, self.img_c) )
            img = img.reshape( (-1, self.img_w*self.img_h*self.img_c) )
            img = img/255.0
            X = np.row_stack( [X, img] )
            y0 = self.get_label(self.pic_idx)
            #print('test last')
            #print(y0)
            Y = np.row_stack( [Y, y0] )
            #print('read data')
            #print(y0)
            self.pic_idx += 1
            if self.pic_idx >= len(self.lst):
                self.pic_idx = 0

        return (X, Y)




class DataSets(object):
    def __init__(self, lst_path, img_w = 227, img_h = 227, img_c = 3, Y_dim=1, train_radio=0.8, test_radio=0.1, valid_radio=0.1):
        if not os.path.exists(lst_path):
            print("%s not exists!" % lst_path)
            return None
        self.lst_path = lst_path
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c
        self.Y_dim = Y_dim
        self.pic_idx = 0
        self.lst = []
        self.train_radio = train_radio
        self.test_radio = test_radio
        self.valid_radio = valid_radio


        self.read_lst()

        random.seed(time.time())
        random.shuffle(self.lst)

        l = self.get_sample_number()
        self.train_lst = self.lst[0: int(self.train_radio*l)]
        self.test_lst = self.lst[int(train_radio*l) : int((self.train_radio+self.test_radio)*l)]
        self.valid_lst = self.lst[int((self.train_radio+self.test_radio)*l) : ]

        self.TrainData = DataReader(self.train_lst, self.img_w, self.img_h, self.img_c, self.Y_dim)
        self.TestData = DataReader(self.test_lst, img_w, img_h, img_c, Y_dim)
        self.ValidData = DataReader(self.valid_lst, img_w, img_h, img_c, Y_dim)


    def read_lst(self):
         for e in open(self.lst_path):
            self.lst.append(e.strip())

    def get_sample_number(self):
        return len(self.lst)
