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

#layertype=0 cnn
#layertype=1 pooling
#filters=32  number of filters
#ksize=3  kernel size
#stride=2  strides
#pad=1  using pad

LayerkeyWords=("layertype=", "filters=", "ksize=", "stride=", "pad=")
LayerVal=(0, 32,3,1)

'''
model的解析类
LayerkeyWords是每一层的关键词
LayerVal存放每一层网络参数
'''
class ModelCFG(object):
    def __init__(self, model_cfg_path):
        self.model_path=model_cfg_path
        self.layers=[]
        
        self.read_model_cfg(model_cfg_path)
        
    
    def read_model_cfg(self, model_cfg_path):
        if not os.path.exists(model_cfg_path):
            print("%s not exists!" % model_cfg_path)
            return None
        
        for line in open(model_cfg_path):
            line=line.strip()
            #print(line)
            LayerVal=[0, 32,3,1, 1]
            
            v=line.split(' ')
            print(v)
            for e in v:
                e = e.strip()
                for i in range(len(LayerkeyWords)):
                    strkey = LayerkeyWords[i]
                    idx = e.find(strkey)
                    if idx >= 0:
                        val=e[len(strkey):]
                        LayerVal[i] = int(val)
            
            # add a layer
            self.layers.append(LayerVal)
            
            
    def print_model_cfg(self):
        print('model file name:', self.model_path)
        
        print("lenth of model=", len(self.layers))
        
        for i in range(len(self.layers)):
            print("layer:", i)
            for j in range(len(LayerkeyWords)):
                strkey = LayerkeyWords[j]
                print(strkey, self.layers[i][j])
                
            print("\n")
            
            
if __name__ == "__main__":
    path = './model_cfg.txt'
    model_cfg = ModelCFG(path)
    model_cfg.print_model_cfg()
