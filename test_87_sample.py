import tensorflow as tf
import numpy as np
import pickle
import cv2
from gcn.models import BjyGCN
from gcn.utils import construct_feed_dict
from gcn.render_tf import render_loss_np
import os
import math
#
#/**************************
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#param
BATCH_SIZE = 5
EPOCH_SIZE = 150
LOAD_FILE_NAME = "../MT_D10000_E100"
LOAD_FLAG = True

img_height = 1024
img_width = 512
channel = 3
all_prefix = []
result_dir = "/root/wcc/iccv/laval_data/sample_param/"
ldr_dir = "/root/wcc/iccv/laval_data/sample_data/"

#light_result_dir ='/root/LightEstimation/ProcessAll/one_light/light_result'
save_dir = './87_sample'

#*******************************/
# Set random seed
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
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')#学习率
flags.DEFINE_integer('epochs',50, 'Number of epochs to train.')#训练次数
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
model_name = "laval_CNNDense105_4GCNRelu_edge127"# % FLAGS.epochs#  "weightsigmoidMask5_CNNDense4GCN_LR6_E100"
model = BjyGCN(placeholders, input_name = model_name, input_dim =3*5, batchsize=FLAGS.batch_size, logging=True)
#
# Load data, initialize session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)
#train_loss = open('true_label_result.txt', 'a')
#
# Runing the demo
feed_dict = construct_feed_dict(placeholders)

pkl_list = []
with open("/root/bjy/BjyFiles/ICCV/laval/Model/127edge_CNNDense105GCN4Relu_mprp/87_sample/file_list.txt", 'r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip()
		pkl_list.append(line)

test_len = 4
kernel_count = 128
batch_size = 5


from gcn.utils_np import render_loss


def output():
    for idx in range(test_len):
        label_group = []
        for i in range(batch_size):
            label_path = result_dir +  pkl_list[int(idx * batch_size + i)] + "_illu.npy"
            label = np.load(label_path)
            label = np.reshape(label, (128, 3))
            label_group.append(label)
        label = np.array(label_group)
        #img = dot_feature
        #img = np.expand_dims(img, 0)
        img_group = []
        origin_img = []
        for i in range(batch_size):
            origin_path = ldr_dir + pkl_list[int(idx * batch_size + i)] + ".png"
            origin_ldr = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(origin_ldr, (FLAGS.input_width, FLAGS.input_height))
            origin_img.append(img)
            img = img / 255.0
            img_group.append(img)
        img = np.array(img_group)
        print(img.shape)

        feed_dict.update({placeholders['img_input']: img})
        feed_dict.update({placeholders['labels']: label})
        #train_loss.write('result %d\n'%(i))
        
        vert = sess.run(model.outputs, feed_dict=feed_dict)
        #vert = np.exp(vert + 1) - 1
        #vert = vert[0]
        #train_loss.write('result %d\n'%(i))
        #print(vert.shape)
        #assert(1==0)
        pre_image = render_loss_np(vert)
        true_image = render_loss_np(label)
        #with tf.Session() as sess:
        #save_path = save_dir + "/" + str(i) + "_gt.png"
        #cv2.imwrite(save_path, image_mid[0]*255)
        print("Now the i is ", idx)
        print("Predict output ", vert.shape, pre_image.shape, true_image[0].shape)
        for w in range(5):
            index = idx * 5 + w
            save_path = save_dir + "/" + str( pkl_list[int(index)]) + "_input.png"
            cv2.imwrite(save_path, origin_img[w])
            save_path = save_dir + "/" + str( pkl_list[int(index)]) + "_gt.exr"
            cv2.imwrite(save_path, true_image[w])
            save_path = save_dir + "/" + str( pkl_list[int(index)]) + "_predict.exr"
            cv2.imwrite(save_path, pre_image[w])
            save_path = save_dir + "/" + str( pkl_list[int(index)]) + "_predict.npy"
            np.save(save_path, vert[w])
            save_path = save_dir + "/" + str( pkl_list[int(index)]) + "_gt.npy"
            np.save(save_path, label[w])
#测试集
output()
test_len = 400
kernel_count = 128
batch_size = 5
Train_Flag = True
if Train_Flag:
    pkl_list = []
    with open("/root/bjy/BjyFiles/LavalTrain/data_helper/laval_train_list_05tail.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            pkl_list.append(line)
    save_dir = "./100epoch/train_test_out"
    test_len = 40
#训练集
#output()

print( 'Finish test')
