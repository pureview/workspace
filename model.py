import numpy as np
import tensorflow as tf
from yolo_cfg import YoloCFG
from model_cfg import ModelCFG


path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)


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


def Model_YOLO(input_image):
    path = './model_cfg.txt'
    model_cfg = ModelCFG(path)
    model_cfg.print_model_cfg()

    img_w=yolo_cfg.get_IMG_W()
    img_h=yolo_cfg.get_IMG_H()
    img_c=yolo_cfg.get_IMG_C()


    x = input_image

    # save lastlayer parameters
    lastlayer = model_cfg.layers[0]

    #"layertype=", "filters=", "ksize=", "stride=", "pad="
    for i in range(len(model_cfg.layers)):
        layer = model_cfg.layers[i]

        if i == 0 and layer[0]==0:
            W_conv = weight_varialbe( [layer[2],layer[2],yolo_cfg.get_IMG_C(),layer[1]], name="conv"+str(i)+"_w")
            b_conv = bias_variable( [layer[1]] , name="bias"+str(i)+"_w")
            x = tf.nn.relu( conv2d(x, W_conv) + b_conv )
            lastlayer = layer
        else:
            if layer[0]==0:
                W_conv = weight_varialbe( [layer[2],layer[2], lastlayer[1], layer[1]], name="conv"+str(i)+"_w")
                b_conv = bias_variable( [layer[1]] , name="bias"+str(i)+"_w")
                x = tf.nn.relu( conv2d(x, W_conv) + b_conv )
                lastlayer = layer

            if layer[0]==1:
                x = max_pool(x, layer[2], layer[3])

    predictions=x

    return predictions



if __name__ == "__main__":
    model = Model_YOLO()
    pass
