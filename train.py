import numpy as np
import os
import cv2
import math
import pickle
#选择对应的CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
#方便训练
from gcn.utils import construct_feed_dict
from gcn.models import BjyGCN
from gcn.fetcher import DataFetcher
from gcn.utils_np import render_loss
import tensorflow as tf

#param setting 
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', 'Data/train_list.txt', 'Data list.')
flags.DEFINE_integer('input_width',360, 'input image width.')#输入图片的宽度
flags.DEFINE_integer('input_height',240, 'input image height.')#输入图片的高度
flags.DEFINE_integer('num_of_light',128, 'number of light sources used.')#输入图片的高度
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')#学习率
flags.DEFINE_integer('epochs',100, 'Number of epochs to train.')#训练次数
flags.DEFINE_integer('batch_size',5, 'Number of epochs to train.')#训练次数
flags.DEFINE_integer('hidden', 128, 'Number of units in GCN hidden layer.') # gcn hidden layer channel中间的维度
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') # image feature dim
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 


# Define placeholders(dict) and model
num_supports = 4
placeholders = {
    'img_input': tf.placeholder(tf.float32, shape=(None, FLAGS.input_height, FLAGS.input_width, 3)),#输入的图像信息
    'labels': tf.placeholder(tf.float32, shape=(None, FLAGS.num_of_light, 3)),#GT的光源信息
    'adj_support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],#邻接表
	'adj_array': tf.placeholder(tf.float32,shape=(128, 128)),
	#'num_features_nonzero' :tf.placeholder(tf.float32),
	'label_mask':tf.placeholder(tf.float32, shape=(None, FLAGS.num_of_light, 1))#用来筛选强度比较大的label
}
#获得pkl 邻接关系完成placeholder

feed_dict = construct_feed_dict(placeholders)
#print("place holder ", placeholders)

#需要修改model dat文件，训练文件的命名，图像的命名
phase = "_0phase_prploss_LR5_E50"
model_name = "laval_CNNDense105_4GCNRelu_edge127"# % FLAGS.epochs#  "weightsigmoidMask5_CNNDense4GCN_LR6_E100"
model = BjyGCN(placeholders, input_name = model_name, input_dim =3*5, batchsize=FLAGS.batch_size, logging=True)
# Load data， 多线程读取数据
data = DataFetcher(FLAGS.data_list)
data.setDaemon(True) #父进程继续执行
data.start()

# Initial session and run
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

# Train graph model
train_loss = open("record_train_loss_" + model_name + ".txt", 'a')
train_loss.write('**************Start training %s***********\n, setting: lr =  %f, Epoch = %d ,hidden = %d\n%s\n'%(
	model_name, FLAGS.learning_rate, FLAGS.epochs, FLAGS.hidden, phase))

#训练的数据量
train_number = data.number
my_all_loss = np.zeros(FLAGS.epochs)
my_param_loss = np.zeros(FLAGS.epochs)
my_render_loss = np.zeros(FLAGS.epochs)
my_percep_loss = np.zeros(FLAGS.epochs)
my_edge_loss = np.zeros(FLAGS.epochs)
#开始训练
#tf.add_to_collection('network-output', model.outputs)
#model.load(sess)
for epoch in range(FLAGS.epochs):
	all_loss = np.zeros(train_number,dtype='float32') 
	all_param_loss = np.zeros(train_number,dtype='float32') 
	all_render_loss = np.zeros(train_number,dtype='float32') 
	all_percep_loss = np.zeros(train_number,dtype='float32') 
	all_edge_loss = np.zeros(train_number,dtype='float32') 
	print("begin epoch" + str(epoch))
	for iters in range(train_number):
		
		
		# Fetch training data
		img_input, y_train, according_label_mask, origin_hdr = data.fetch()
		feed_dict.update({placeholders['img_input']: img_input})
		feed_dict.update({placeholders['labels']: y_train})
		feed_dict.update({placeholders['label_mask']:according_label_mask})

		# Training step
		#vert = sess.run(model.outputs, feed_dict=feed_dict)
		#vert = sess.run(model.outputs, feed_dict=feed_dict)
		_, modelloss,paramloss,renderloss,perceploss,edgeloss, out1 = sess.run([model.opt_op,model.loss,model.paramloss, model.renderloss, model.perceploss, model.edge_loss, model.outputs], feed_dict=feed_dict)
		#modelloss,paramloss,renderloss,perceploss,out1 = sess.run([ model.loss,model.paramloss, model.renderloss, model.perceploss, model.outputs], feed_dict=feed_dict)
		
		'''print(model.loss)
		with tf.GradientTape() as gen_tape:
			generator_gradients = gen_tape.gradient((model.loss),
											model.vars)
			
			g_optimizer.apply_gradients(zip(generator_gradients,
											model.vars))
        #d_opt_op = d_optimizer.minimize(d_loss)'''
		#print("in 85line ", out1)
		if iters == 256 or iters == 1200:
			pre_image = render_loss(out1)
			true_image = render_loss(y_train)
			print('here to save training out  Epoch %d, Iteration %d'%(epoch,iters))
			for w in range(5):
				#out_label = out1[w]
				#pre_image = render_loss(out1[w])
				#true_image = render_loss(y_train[w])
				
				save_path = "./training_out/Iter" + str(iters) + "_index" + str(w) +"_E"+ str(epoch)  +  "_gt.exr"
				if epoch == 0:
					cv2.imwrite(save_path, true_image[w].eval(session=sess))
				save_path ="./training_out/Iter" + str(iters) + "_index" + str(w) +"_E"+ str(epoch) + "_test_show.exr"
				cv2.imwrite(save_path, pre_image[w].eval(session=sess))
		#存loss
		all_loss[iters] = modelloss
		all_param_loss[iters] = paramloss
		all_render_loss[iters] = renderloss
		all_percep_loss[iters] = perceploss
		all_edge_loss[iters] = edgeloss
		mean_loss = np.mean(all_loss[np.where(all_loss)])
		mean_paramloss = np.mean(all_param_loss[np.where(all_param_loss)])
		mean_renderloss = np.mean(all_render_loss[np.where(all_render_loss)])
		mean_perceploss = np.mean(all_render_loss[np.where(all_percep_loss)])
		if (iters+1) % 128 == 0:
			#print("in 101line ", all_loss)
			print('Epoch %d, Iteration %d'%(epoch + 1,iters + 1))
			print('Mean loss = %f, param loss = %f, render loss = %f, percep loss = %f, edge loss = %f,  model loss = %f, %d'%(
					mean_loss,paramloss, renderloss, perceploss, edgeloss, modelloss, data.queue.qsize()))
	# Save model
	#model_path = "./model/" + model_name + "train.ckpt"
	#saver = tf.train.Saver()
	#saver.save(sess, model_path)
	model.save(epoch, sess)
	mean_loss = np.mean(all_loss)
	my_all_loss[epoch] = mean_loss
	mean_param_loss = np.mean(all_param_loss)
	my_param_loss[epoch] = mean_param_loss
	mean_render_loss = np.mean(all_render_loss)
	my_render_loss[epoch] = mean_render_loss
	mean_percep_loss = np.mean(all_percep_loss)
	my_percep_loss[epoch] = mean_percep_loss
	mean_edge_loss = np.mean(all_edge_loss)
	my_edge_loss[epoch] = mean_edge_loss
	train_loss.write('Epoch %d, loss %f, param loss %f render loss %f percep loss %f\n'%(
					epoch+1, mean_loss, mean_param_loss,mean_render_loss, mean_percep_loss))
	
	train_loss.flush()
#保存文件
import matplotlib.pyplot as plt
x = np.arange(1,FLAGS.epochs+1,1)
plt.plot(x, my_all_loss, color='red', label='all loss')
plt.plot(x, my_param_loss, color='green', label='param loss')
plt.plot(x, my_render_loss*0.2, color='blue', label='0.2*render loss')
plt.plot(x, my_percep_loss*0.1, color='black', label='0.1*percep loss')
#plt.plot(x, my_edge_loss*0.01, color='black', label='0.01*edge loss')
# plt.plot(history.history['val_accuracy'])
plt.legend()
plt.savefig("loss_" + model_name + str(FLAGS.epochs) + phase + ".png")

data.shutdown()
train_loss.close()
print('Training Finished!')
