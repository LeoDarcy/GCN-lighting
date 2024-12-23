#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import tensorflow as tf
import math
resize_width = 128
resize_height = 64
ori_width = 128
ori_height = 64
kernel_count = 128
input_height = 240
input_width = 320
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
#测试实践
def cal_I_pre(x, alpha, elevation, azimuth, weight):
    
    left = tf.sin(elevation) * (tf.reshape(tf.sin(x[:, :, 0]), [-1, 1]))
    #print("lefdt",left)
    right1 = tf.tile(tf.reshape(x[:, :, 1], [-1, 1]), [1, kernel_count])
    #print("right1", right1)
    right1 = tf.cos(right1 - azimuth)
    #print("right1", right1)
    right2 = tf.reshape(tf.cos(x[:, :, 0]), [-1, 1]) * tf.cos(elevation)
    # print("right2",right2)
    # right = tf.reshape(tf.cos(x[:,:,0]), [-1,1])*tf.cos(elevation)*tf.cos(tf.tile(tf.reshape(x[:,1],[-1,1]),[1,3])-azimuth)
    right = right2 * right1
    # print("right", right)
    mi = left + right - 1
    mi = tf.debugging.check_numerics(mi, "pre mi non number")
    exp = tf.exp(mi * alpha)

    weight = tf.expand_dims(weight, 1)
    I = tf.stack((exp, exp, exp), axis=-1) * weight
    I = tf.reduce_sum(I, axis=2)
    return I
#计算128个光源的方位角和仰角
def get_center_elevation_azimuth(n):
    angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = angle * np.arange(n, dtype=np.float32)
    y = np.linspace(1.0 - 1.0 / n, 1.0 / n - 1.0, n)
    center = np.zeros((n, 2), dtype=np.float32)
    #print(math.asin(y))
    tmp = [math.asin(x) for x in y]
    #print("an", y, tmp)
    center[:, 0] = [math.asin(x) for x in y]
    center[:, 1] = theta
    #print("herh", center[:, 0])
    return center
BATCH_SIZE = 1
pic_alpha = np.zeros((kernel_count), dtype=np.float32)
pic_alpha[:] = math.log(0.5) / (math.cos(math.atan(2.0 / math.sqrt(kernel_count))) - 1.0)
pic_center = get_center_elevation_azimuth(kernel_count)

pic_elevation = pic_center[:,0]
pic_azimuth = pic_center[:, 1]
pic_elevation = np.expand_dims(pic_elevation, 0)
pic_elevation = np.expand_dims(pic_elevation, 0)
pic_elevation = np.reshape(pic_elevation, (1,1,128))
pic_elevation = np.tile(pic_elevation, (BATCH_SIZE, 1,1))

pic_azimuth = np.reshape(pic_azimuth, (1,1,128))
pic_azimuth = np.tile(pic_azimuth, (BATCH_SIZE, 1, 1))
def render_loss(y_pre):
    #y_true是一个图像，y_pre是一个128*3的参数
    alpha = pic_alpha
    center = pic_center
    elevation = pic_elevation
    azimuth = pic_azimuth
    weights = tf.reshape(y_pre,  (-1, 128, 3))
    x = np.expand_dims(data_re_map, 0)
    pre_I = cal_I_pre(x, alpha, elevation, azimuth, weights)
    print("In util_np line 107", pre_I.shape)
    pre_I = tf.reshape(pre_I, (-1, resize_height, resize_width, 3))
    return pre_I