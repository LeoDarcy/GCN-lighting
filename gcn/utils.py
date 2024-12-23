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
import pickle
def construct_feed_dict(placeholders):
    pkl = pickle.load(open('/root/bjy/BjyFiles/LavalTrain/Dataset/edgefile/bjy_Edge127.dat', 'rb'))
    """Construct feed dictionary."""
    #pkl的结构：
    #len = 2, 0是坐标，1是邻接的边
    coord = pkl[0]
    feed_dict = dict()
    feed_dict.update({placeholders['adj_support'][i]: pkl[1][i] for i in range(len(pkl[1]))})

    pkl = pickle.load(open('/root/bjy/BjyFiles/LavalTrain/Dataset/edgefile/bjy_Edge127.dat', 'rb'))
    feed_dict.update({placeholders['adj_support'][2 + i]: pkl[1][i] for i in range(len(pkl[1]))})
    #feed_dict.update({placeholders['num_features_nonzero']: 3})

    
    adj_array = np.load("/root/bjy/BjyFiles/LavalTrain/Dataset/edgefile/bjy_Edge10_adj.npy")
    adj_array = np.array(adj_array)
    feed_dict.update({placeholders['adj_array']: adj_array})
    return feed_dict
