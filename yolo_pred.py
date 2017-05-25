import numpy as np
import tensorflow as tf
from yolo_cfg import YoloCFG
from models import Model_YOLO20
import math


path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)


def get_ret_loc(ret):
    conf = ret[:,:,:,4]
    final=[]
    for i in range(len(conf)):
        max_conf=-1.0
        max_index=[0,0]
        for j in range(len(conf[i])):
            for k in range(len(conf[i][j])):
                if conf[i][j][k]>max_conf:
                    max_conf=conf[i][j][k]
                    max_index=[j,k]
        final.append(max_index)
    return np.array(final)



def yolo_predictions(y_pred):

    GRID_DEPT=6
    y_pred0 = tf.reshape(y_pred, [-1, yolo_cfg.get_S()*yolo_cfg.get_S(), GRID_DEPT])

    ret = tf.argmax(y_pred0, axis=1)

    return ret

