
import numpy as np
import tensorflow as tf
import math
resize_width = 128
resize_height = 64
ori_width = 128
ori_height = 64
kernel_count = 128
input_height = 240
input_width = 360
model_alpha = 0.1
row_cat = 6
col_cat = 6
def ae_to_xy(azimuth, elevation):
    elevation = np.tanh(elevation) * math.pi/2
    azimuth = np.tanh(azimuth) * math.pi
    y = (azimuth + math.pi) / math.pi / 2 * ori_width
    x = ori_height - 1 - (elevation + math.pi / 2) / math.pi * ori_height
    return x, y
def xy_to_omega(x, y, height, width):
    half_width = width / 2
    half_height = height / 2
    x = (x - half_width + 0.5) / half_width
    y = (y - half_height + 0.5) / half_height
    return (y * np.pi * 0.5, x * np.pi)
data_re_map = np.zeros((resize_width * resize_height, 2), dtype=np.float32)
data_idx = 0
for y in range(resize_height):
    for x in range(resize_width):
        omega = xy_to_omega(x, resize_height - 1 - y, resize_height, resize_width)
        data_re_map[data_idx] = (omega[0], omega[1])
        data_idx += 1

def get_center(n):
    angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = angle * np.arange(n, dtype=np.float32)
    y = np.linspace(1.0 - 1.0 / n, 1.0 / n - 1.0, n)
    r = np.sqrt(1.0 - y * y)
    center = np.zeros((n, 3), dtype=np.float32)
    center[:, 0] = r * np.cos(theta)
    center[:, 1] = y
    center[:, 2] = r * np.sin(theta)
    return center
def eval_tensorflow(x, weight, alpha, center):
    #left = tf.sin(elevation) * (tf.reshape(tf.sin(x[:, :, 0]), [-1, 1]))
    #x是xy_omega
    #print("in eval_tensorflow",(np.cos(x[:,0]) * np.sin(x[:,1])).shape)
    mi = center[:, 0] * np.reshape((np.cos(x[:,0]) * np.sin(x[:,1])), (-1,1))
    #print("mi ", mi.shape)
    mi = mi + center[:, 1] * np.reshape(np.sin(x[:,0]), (-1,1))
    #print("mi ", mi.shape)
    mi = mi + -center[:, 2] * np.reshape((np.cos(x[:,0]) * np.cos(x[:,1])), (-1,1)) - 1.0
    #print("mi ", mi.shape)
    exp = np.exp(alpha * mi)
    #print("exp ", exp.shape)
    #exp = tf.exp(mi * alpha)

    weight = tf.expand_dims(weight, 1)
    #BGR sequence
    I = tf.stack((exp, exp, exp), axis=-1)
    #print("I ", I.shape)
    I = I * weight
    #print("I ", I.shape)
    #I = tf.reduce_sum(I, axis=1)
    #I = (weight.T * exp).T
    return tf.reduce_sum(I, axis=2)
def render_loss_tensorflow(y_pre):
    alpha = np.zeros((kernel_count), dtype=np.float32)
    alpha[:] = math.log(0.6) / (math.cos(math.atan(2.0 / math.sqrt(kernel_count))) - 1.0)
    center = get_center(kernel_count)
    weight = tf.reshape(y_pre,  ( -1, 128, 3))
    I = np.zeros((resize_height, resize_width, 3), dtype=np.float32)
    I = eval_tensorflow(data_re_map, weight, alpha, get_center(128))
    print("End render loss_tensorflow in util ", I.shape)
    #assert(1==0)
    return I
    

def eval_tensorflow_np(x, weight, alpha, center):
    #left = tf.sin(elevation) * (tf.reshape(tf.sin(x[:, :, 0]), [-1, 1]))
    #x是xy_omega
    #print("in eval_tensorflow",(np.cos(x[:,0]) * np.sin(x[:,1])).shape)
    mi = center[:, 0] * np.reshape((np.cos(x[:,0]) * np.sin(x[:,1])), (-1,1))
    #print("mi ", mi.shape)
    mi = mi + center[:, 1] * np.reshape(np.sin(x[:,0]), (-1,1))
    #print("mi ", mi.shape)
    mi = mi + -center[:, 2] * np.reshape((np.cos(x[:,0]) * np.cos(x[:,1])), (-1,1)) - 1.0
    #print("mi ", mi.shape)
    exp = np.exp(alpha * mi)
    #print("exp ", exp.shape)
    #exp = tf.exp(mi * alpha)

    weight = np.expand_dims(weight, 1)
    #print("line 60, weight shape", weight.shape)
    #BGR sequence
    I = np.stack((exp, exp, exp), axis=-1)
    #print("line 99 I ", I.shape)
    I = I * weight
    #print("I ", I.shape)
    #I = tf.reduce_sum(I, axis=1)
    #I = (weight.T * exp).T
    return np.sum(I, axis=2)
def render_loss_np(y_pre):
    alpha = np.zeros((kernel_count), dtype=np.float32)
    alpha[:] = math.log(0.6) / (math.cos(math.atan(2.0 / math.sqrt(kernel_count))) - 1.0)
    center = get_center(kernel_count)
    weight = np.reshape(y_pre,  (-1, 128, 3))
    I = np.zeros((resize_height, resize_width, 3), dtype=np.float32)
    I = eval_tensorflow_np(data_re_map, weight, alpha, get_center(128))
    #print("line 112 ", I.shape)
    I = np.reshape(I, (-1, resize_height, resize_width, 3))
    print("End render loss_tensorflow in util ", I.shape)
    return I