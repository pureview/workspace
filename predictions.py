import numpy as np
import tensorflow as tf
from yolo_cfg import YoloCFG
from model_cfg import ModelCFG
from model import Model_YOLO


path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)


def predictions(pred):
    # get xy
    y_pred_xy = pred[:,:,:,0:2]
    
    # get wh
    y_pred_wh = pred[:,:,:,2:4]

    
    y_pred_conf = pred[:,:,:,4]

    y_pred_class = pred[:,:,:,5]
    
    tf.maxarg(y_pred_conf)
    

if __name__ == "__main__":
    #model = Model_YOLO()
    predictions()
    pass
