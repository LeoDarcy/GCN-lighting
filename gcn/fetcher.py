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
import pickle
import threading
import queue
import sys
import cv2
import os
import tensorflow as tf
#from skimage import io,transform
flags = tf.app.flags
FLAGS = flags.FLAGS
kernel_count = 128
input_height = 240
input_width = 360
result_dir = "/root/wcc/iccv/laval_data/sample_param/"
ldr_dir = "/root/wcc/iccv/laval_data/sample_data/"
batch_size = 5
data_len = int(10000 / batch_size)
#data_len = 2
resize_width = 128
resize_height = 64
normal_zone = 5#表示归一化到【-5，5】之间


#pkl_file = open('/root/LightEstimation/sun360_128weights/NumName.pkl', 'rb')

#NumNameDict = pickle.load(pkl_file)
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def ACESFilm(x):
		a = 2.51
		b = 0.03
		c = 2.43
		d = 0.59
		e = 0.14
		saturate = 1.0
		result = saturate * ((x*(a*x+b))/(x*(c*x+d)+e))
		#print("result", result.shape, result)
		return result
class DataFetcher(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher, self).__init__()
		self.stopped = False
		self.queue = queue.Queue(64)
		#file_list是读取的文件目录
		self.pkl_list = []
		with open("/root/bjy/BjyFiles/LavalTrain/data_helper/laval_train_list_05tail.txt", 'r') as f:
			count = 0
			while(count < data_len * batch_size):
				count += 1
				line = f.readline().strip()
				if not line:
					break
				self.pkl_list.append(line)
		
		#self.pkl_list = NumNameDict
		#self.pkl_list = None
		self.index = 0
		self.number = data_len
		np.random.shuffle(self.pkl_list)

		#self.npy_img = np.load("/root/LightEstimation/sun360_128weights/wangfilter/wangfilter_traindataset_10000.npy")
		#self.npy_label = np.load("/root/LightEstimation/sun360_128weights/wangfilter/wangfilter_trainlabel_10000.npy")
		#self.npy_feature = np.load("/root/LightEstimation/MyNetV2/10000_feature.npy")

	def work(self, idx):
		'''pkl_path = self.pkl_list[idx]
		#读取origin hdr，转换为小图
		origin_path = origin_hdr_path + "/" + pkl_path + ".exr"
		assert(os.path.isfile(origin_hdr_path + "/" + pkl_path + ".exr"))
		origin_hdr = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
		origin_hdr = cv2.resize(origin_hdr, (resize_width, resize_height))'''
		origin_hdr = np.zeros(((64, 128, 3)))
		#origin_hdr = np.reshape(origin_hdr, (-1, 3))
		#print("here in fetcher",img.shape)
		#pre_weight = layer_model.predict(img)
		#pre_weight = np.reshape(pre_weight[0], (128, 3))
		label_group = []
		for i in range(batch_size):
			label_path = result_dir +  self.pkl_list[int(idx * batch_size + i)] + "_illu.npy"
			label = np.load(label_path)
			#print(label.shape)
			#assert(1==0)
			label = np.reshape(label, (128, 3))
			label_group.append(label)
		label = np.array(label_group)
		#img = dot_feature
		#img = np.expand_dims(img, 0)
		img_group = []
		for i in range(batch_size):
			origin_path = ldr_dir + "/" + self.pkl_list[int(idx * batch_size + i)] + ".png"
			origin_ldr = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
			img = cv2.resize(origin_ldr, (input_width, input_height))
			img = img / 255.0
			img_group.append(img)
		img = np.array(img_group)
		#构建mask
		mask = np.ones((batch_size, label.shape[0], 1))
		b_weight = label[:, :, 0]
		g_weight = label[:, :, 1]
		r_weight = label[:, :, 2]
		light_weight = r_weight * 0.3 + g_weight * 0.59 + b_weight * 0.11
		'''light_weight = (light_weight - np.min(light_weight)) / (np.max(light_weight) - np.min(light_weight))
		light_weight = -normal_zone + 2 * normal_zone * light_weight
		mask = sigmoid(light_weight)'''
		
		mask = ACESFilm(light_weight)
		
		#print(mask.shape)(5,128)
		#assert(1==0)
		mask = np.reshape(mask, (batch_size, 128,1))
		#mask = np.zeros((128,1))
		#label = np.log(1 + label)
		#mylist = [0, 5, 8, 13, 16, 18, 21, 26, 29, 34, 39, 42, 47, 50, 55, 60, 63, 68, 71, 73, 76, 81, 84, 89, 94, 97, 102, 105, 110, 115, 118, 123]
		#mask[mylist] = 0
		return img, label, mask, origin_hdr
	
	def run(self):
		while self.index < 999999999 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

if __name__ == '__main__':
	file_list = sys.argv[1]
	data = DataFetcher(file_list)
	data.start()
	for i in range(99999):
		image,point,path = data.fetch()
		print("in main " + str(i))
		print(image.shape)
		print(point)
		print(path)
		print(point.shape)
		print(path.shape)
		assert(1==0)
	data.stopped = True
