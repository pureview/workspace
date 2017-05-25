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
import numpy as np


keyWords=("S=", "coords=", "object_scale=", "noobject_scale=", "class_scale=", "coord_scale=", "input_W=", "input_H=", "input_C=")

'''
yolo cfg 文件解析类
keyWords是yolo cfg文件的关键词
keyWordsVal存放yolo算法的关键参数
'''
class YoloCFG(object):
    def __init__(self, yolo_cfg_path):
        
        self.cfg_path=yolo_cfg_path
        self.keyWordsVal=len(keyWords)*['']        
        self.read_yolo_cfg(yolo_cfg_path)
        
    
    def read_yolo_cfg(self, yolo_cfg_path):
        if not os.path.exists(yolo_cfg_path):
            print("%s not exists!" % yolo_cfg_path)
            return None
        
        for line in open(yolo_cfg_path):
            line=line.strip()
            #print(line)
            for i in range(len(keyWords)):
                strkey = keyWords[i]
                if line.find(strkey) == 0:
                    val=line[len(strkey):]
                    self.keyWordsVal[i] = float(val)
            
    def print_yolo_cfg(self):
        print('cfg file name:', self.cfg_path)
        for i in range(len(keyWords)):
            strkey = keyWords[i]
            print(strkey, self.keyWordsVal[i])
            
    def get_S(self):
        return int(self.keyWordsVal[0])
    
    def get_coords(self):
        return int(self.keyWordsVal[1])
    
    def get_object_scale(self):
        return float(self.keyWordsVal[2])
    
    def get_noobject_scale(self):
        return float(self.keyWordsVal[3])
    
    def get_class_scale(self):
        return float(self.keyWordsVal[4])
    
    def get_coord_scale(self):
        return float(self.keyWordsVal[5])
    
    def get_IMG_W(self):
        return int(self.keyWordsVal[6])
    
    def get_IMG_H(self):
        return int(self.keyWordsVal[7])
    
    def get_IMG_C(self):
        return int(self.keyWordsVal[8])
            
            
if __name__ == "__main__":
    path = './yolo_cfg.txt'
    cfg = YoloCFG(path)
    cfg.print_yolo_cfg()
