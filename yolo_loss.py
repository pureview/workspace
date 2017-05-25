# -*- coding: utf-8 -*-

from __future__ import print_function
import os,sys,math
import numpy as np
import tensorflow as tf
from yolo_cfg import YoloCFG


path = './yolo_cfg.txt'
yolo_cfg = YoloCFG(path)



#shape: N*S*S*(6) 6=1*(4+1)+1
def yolo_loss_0(y_pred, y_true):
    """
    yolo_loss
    """

    y_true_conf = y_true[:,:,:,4]
    y_pred_conf = y_pred[:,:,:,4]

    # scale
    object_scale = yolo_cfg.get_object_scale()
    no_object_scale = yolo_cfg.get_noobject_scale()
    class_scale = yolo_cfg.get_class_scale()
    coordinates_scale = yolo_cfg.get_coord_scale()


    # no_objects_loss
    no_objects_loss = no_object_scale*tf.square(0-y_pred_conf)*(1 - y_true_conf)
    # objects_loss
    objects_loss = object_scale*tf.square(1-y_pred_conf)*y_true_conf


    # confidence_loss
    confidence_loss = objects_loss + no_objects_loss
    conf_loss_sum = tf.reduce_mean(confidence_loss)

    return conf_loss_sum






#shape: N*S*S*(6) 6=1*(4+1)+1
def yolo_loss_debug(y_pred, y_true):
    """
    yolo_loss
    """

    # scale
    object_scale = yolo_cfg.get_object_scale()
    no_object_scale = yolo_cfg.get_noobject_scale()
    class_scale = yolo_cfg.get_class_scale()
    coordinates_scale = yolo_cfg.get_coord_scale()

    # get xy
    y_true_xy = y_true[:,:,:,0:2]
    y_pred_xy = y_pred[:,:,:,0:2]

    # get wh
    y_true_wh0 = y_true[:,:,:,2:4]
    y_pred_wh0 = y_pred[:,:,:,2:4]

    y_true_wh = y_true[:,:,:,2:4]
    y_pred_wh = y_pred[:,:,:,2:4]


    #################################
    # get wh square
    #y_true_wh = tf.sqrt(y_true_wh)
    #y_pred_wh = tf.sqrt(y_pred_wh)

    # predict sqrt(w) sqrt(h)
    y_true_wh = tf.square(y_true_wh)
    y_pred_wh = tf.square(y_pred_wh)


    y_true_conf = y_true[:,:,:,4]
    y_pred_conf = y_pred[:,:,:,4]

    y_true_class = y_true[:,:,:,5]

    y_pred_class = y_pred[:,:,:,5]


    # calc loss
    # coords loss
    # (x-x')^2+(y-y')^2
    xy_square = tf.square(y_pred_xy - y_true_xy)
    coord_loss_xy = tf.reduce_sum(xy_square, axis=3)*y_true_conf


    # (ww-ww')^2+(hh-hh')^2
    wh_square = tf.square(y_pred_wh0 - y_true_wh0)
    coord_loss_wh = tf.reduce_sum(wh_square, axis=3)*y_true_conf

    # coord_loss
    coord_loss = coordinates_scale*(coord_loss_xy+coord_loss_wh)*y_true_conf
    coord_loss_sum = tf.reduce_sum(coord_loss)


    # scale
    object_scale = yolo_cfg.get_object_scale()
    no_object_scale = yolo_cfg.get_noobject_scale()
    class_scale = yolo_cfg.get_class_scale()
    coordinates_scale = yolo_cfg.get_coord_scale()


    # class_loss loss
    class_loss = class_scale*tf.square(y_pred_class - y_true_class)*y_true_conf
    class_loss_sum = tf.reduce_sum(class_loss)

    # loss0
    loss0 = tf.reduce_sum( tf.square(y_pred_conf - y_true_conf)*y_true_conf )


    return coord_loss_sum+class_loss_sum+loss0




#shape: N*S*S*(6) 6=1*(4+1)+1
def yolo_loss(y_pred, y_true):
    """
    yolo_loss
    """
    # get xy
    y_true_xy = y_true[:,:,:,0:2]
    y_pred_xy = y_pred[:,:,:,0:2]

    # get wh
    y_true_wh0 = y_true[:,:,:,2:4]
    y_pred_wh0 = y_pred[:,:,:,2:4]

    y_true_wh = y_true[:,:,:,2:4]
    y_pred_wh = y_pred[:,:,:,2:4]


    #################################
    # get wh square
    #y_true_wh = tf.sqrt(y_true_wh)
    #y_pred_wh = tf.sqrt(y_pred_wh)

    # predict sqrt(w) sqrt(h)
    # 真实宽度和高度
    y_true_wh = tf.square(y_true_wh)
    y_pred_wh = tf.square(y_pred_wh)

    # 物体置信度和分类置信度
    y_true_conf = y_true[:,:,:,4]
    y_pred_conf = y_pred[:,:,:,4]

    y_true_class = y_true[:,:,:,5]
    y_pred_class = y_pred[:,:,:,5]

    


    # calc loss
    # coords loss
    # (x-x')^2+(y-y')^2
    xy_square = tf.square(y_pred_xy - y_true_xy)
    # 坐标损失函数
    coord_loss_xy = tf.reduce_sum(xy_square, axis=3)*y_true_conf


    # (ww-ww')^2+(hh-hh')^2
    # 宽高损失函数
    wh_square = tf.square(y_pred_wh0 - y_true_wh0)
    coord_loss_wh = tf.reduce_sum(wh_square, axis=3)*y_true_conf


    # iou loss
    pred_wh_half = y_pred_wh / 2.
    pred_mins = y_pred_xy - pred_wh_half
    pred_maxes = y_pred_xy + pred_wh_half

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = y_true_wh / 2.
    true_mins = y_true_xy - true_wh_half
    true_maxes = y_true_xy + true_wh_half


    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_mins = tf.maximum(intersect_mins, 0)

    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_maxes = tf.minimum(intersect_maxes, 1.0)

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)

    intersect_areas = intersect_wh[:,:,:, 0] * intersect_wh[:,:,:, 1]
    
    #############################################################
    tf.clip_by_value(intersect_areas, 0.0, 1.0)
    #############################################################


    pred_areas = y_pred_wh[..., 0] * y_pred_wh[..., 1]
    true_areas = y_true_wh[..., 0] * y_true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    union_areas = tf.maximum(union_areas, 1e-10)
    # todo: 这里应该对每一个格子计算IOU
    # note: 这里intersect_areas和y_true_conf都是 3-d 张量[batch,x_index,y_index]
    intersect_areas = intersect_areas*y_true_conf
    union_areas = union_areas*y_true_conf
    # 1e-5 太大
    iou_scores = tf.truediv(intersect_areas, union_areas)
    iou_scores = tf.clip_by_value(iou_scores, 0.0, 1.0)
    #iou_scores = tf.truediv(intersect_areas, (union_areas+1.0E-12))



    # scale
    object_scale = yolo_cfg.get_object_scale()
    no_object_scale = yolo_cfg.get_noobject_scale()
    class_scale = yolo_cfg.get_class_scale()
    coordinates_scale = yolo_cfg.get_coord_scale()


    # no_objects_loss
    #no_objects_loss = no_object_scale*tf.square(0.0-y_pred_conf)*(1.0 - y_true_conf)
    # objects_loss
    #objects_loss = object_scale*tf.square(1.0-y_pred_conf)*y_true_conf + tf.square(iou_scores-y_pred_conf)*y_true_conf

    # 这里只对有物体的格子求loss
    y_true_conf_IOU = y_true_conf * iou_scores
    # no_objects_loss
    no_objects_loss = no_object_scale*tf.square(y_pred_conf - 0.0)*(1.0 - y_true_conf)
    # objects_loss
    objects_loss = object_scale*tf.square(y_pred_conf - y_true_conf_IOU)*y_true_conf



    # iou loss
    #iou_loss_sum= tf.reduce_sum(tf.square(1.0-iou_scores))

    # confidence_loss
    #confidence_loss = objects_loss + no_objects_loss
    #conf_loss_sum = tf.reduce_sum(confidence_loss)
    conf_loss_sum = tf.reduce_sum(objects_loss) + tf.reduce_sum(no_objects_loss)

    # class_loss loss
    class_loss = class_scale*tf.square(y_pred_class - y_true_class)*y_true_conf
    class_loss_sum = tf.reduce_sum(class_loss)

    # coord_loss
    coord_loss = coordinates_scale*(coord_loss_xy+coord_loss_wh)*y_true_conf
    coord_loss_sum = tf.reduce_sum(coord_loss)



    # total_loss
    #debug_iou_loss = 0.5*(iou_loss_sum)
    debug_conf_loss = 0.5*(conf_loss_sum)
    debug_coord_loss = 0.5*(coord_loss_sum)
    debug_class_loss = 0.5*(class_loss_sum)

    #total_loss = debug_iou_loss+ debug_conf_loss + debug_coord_loss + debug_class_loss
    #total_loss = debug_iou_loss + debug_conf_loss + debug_coord_loss + debug_class_loss
    #total_loss = debug_conf_loss + debug_coord_loss + debug_class_loss
    #total_loss = debug_conf_loss
    total_loss = debug_conf_loss + debug_coord_loss + debug_class_loss
    #total_loss = tf.reduce_sum(objects_loss)


    return total_loss
    return tf.reduce_sum(tf.square(y_pred_conf - y_true_conf))/32.0




#shape: N*S*S*(6) 6=1*(4+1)+1
def get_iou(y_pred, y_true):
    """
    yolo_loss
    """
    # get xy
    y_true_xy = y_true[:,:,:,0:2]
    y_pred_xy = y_pred[:,:,:,0:2]

    # get wh
    y_true_wh0 = y_true[:,:,:,2:4]
    y_pred_wh0 = y_pred[:,:,:,2:4]

    y_true_wh = y_true[:,:,:,2:4]
    y_pred_wh = y_pred[:,:,:,2:4]


    #################################
    # get wh square
    #y_true_wh = tf.sqrt(y_true_wh)
    #y_pred_wh = tf.sqrt(y_pred_wh)

    # predict sqrt(w) sqrt(h)
    y_true_wh = tf.square(y_true_wh)
    y_pred_wh = tf.square(y_pred_wh)


    y_true_conf = y_true[:,:,:,4]
    y_pred_conf = y_pred[:,:,:,4]

    # iou loss
    pred_wh_half = y_pred_wh / 2.
    pred_mins = y_pred_xy - pred_wh_half
    pred_maxes = y_pred_xy + pred_wh_half

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = y_true_wh / 2.
    true_mins = y_true_xy - true_wh_half
    true_maxes = y_true_xy + true_wh_half


    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_mins = tf.maximum(intersect_mins, 0)

    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_maxes = tf.minimum(intersect_maxes, 1.0)

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)

    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    #############################################################
    tf.clip_by_value(intersect_areas, 0.0, 1.0)
    #############################################################


    pred_areas = y_pred_wh[..., 0] * y_pred_wh[..., 1]
    true_areas = y_true_wh[..., 0] * y_true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    union_areas = tf.maximum(union_areas, 1e-10)

    intersect_areas = tf.reduce_sum(intersect_areas*y_true_conf)
    union_areas = tf.reduce_sum(union_areas*y_true_conf)
    iou_scores = tf.truediv(intersect_areas, (union_areas+0.00001))
    iou_scores = tf.clip_by_value(iou_scores, 0.0, 1.0)
    #iou_scores = tf.truediv(intersect_areas, (union_areas+1.0E-12))

   
    # iou loss
    #iou_loss = iou_scores*y_true_conf
    #iou_loss_sum = tf.reduce_sum(iou_loss)
    iou_loss_sum= iou_scores


    # total_loss
    debug_iou_loss = 0.5*(iou_loss_sum)

    return debug_iou_loss


#shape: N*S*S*(6) 6=1*(4+1)+1
def get_conf(y_pred, y_true):
    """
    yolo_loss
    """

    y_true_conf = y_true[:,:,:,4]
    y_pred_conf = y_pred[:,:,:,4]


    # scale
    object_scale = yolo_cfg.get_object_scale()
    no_object_scale = yolo_cfg.get_noobject_scale()
    class_scale = yolo_cfg.get_class_scale()
    coordinates_scale = yolo_cfg.get_coord_scale()


    # no_objects_loss
    no_objects_loss = no_object_scale*tf.square(0.0-y_pred_conf)*(1.0 - y_true_conf)
    # objects_loss
    objects_loss = object_scale*tf.square(1.0-y_pred_conf)*y_true_conf

    # confidence_loss
    #confidence_loss = objects_loss + no_objects_loss
    #conf_loss_sum = tf.reduce_sum(confidence_loss)
    conf_loss_sum = tf.reduce_sum(objects_loss) + tf.reduce_sum(no_objects_loss)


    # total_loss
    #debug_iou_loss = 0.5*(iou_loss_sum)
    debug_conf_loss = 0.5*(conf_loss_sum)

    return debug_conf_loss/16.0




def make_grids(S):
    SS = np.zeros( (S,S) )

    return SS




def test():
    S = yolo_cfg.get_S()
    GRID_DEPT=6

    true_grid = [0.5, 0.3, math.sqrt(0.4), math.sqrt(0.5), 1.0, 1.0]
    true = np.zeros( (1,S,S,GRID_DEPT) )
    for i in range(S):
        for j in range(S):
            true[0,i,j,:] = true_grid
    true = np.reshape(true, (1,S,S,GRID_DEPT))

    var_true = tf.constant(true)


    pred_grid = [0.6, 0.4, math.sqrt(0.3), math.sqrt(0.2), 1.0, 1.0]
    pred = np.zeros( (1,S,S,GRID_DEPT) )
    for i in range(S):
        for j in range(S):
            pred[0,i,j,:] = pred_grid
    pred = np.reshape(pred, (1,S,S,GRID_DEPT))

    var_pred = tf.constant(pred)



    loss = yolo_loss(var_true, var_pred)
    print("loss=", loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    l = sess.run(loss)
    #print(l[0,0,:])
    print(l)
    print(l.shape)



def get_loc(x, y):
    (loc_x, loc_y) = (-1, -1)

    S = yolo_cfg.get_S()

    x = x*S
    y = y*S


    GRID_DEPT=6 #6=1*(4+1)+1

    loc_x = int(x)
    loc_y = int(y)

    gx = x-loc_x
    gy = y-loc_y

    return (loc_x, loc_y, gx, gy)


if __name__ == "__main__":

    a = np.zeros( [2,3,3,6] )
    b = np.zeros( [2,3,3,6] )
    a[0,1,1,0:6] = [0.4,0.4,0.2,0.2,1.0,1.0]
    b[0,1,1,0:6] = [0.44,0.44,0.24,0.24,1.0,1.0]
    pass
