import tensorflow as tf
import numpy as np
import scipy.io as sio
_vgg_params = None

def vgg_params():
    global _vgg_params
    if _vgg_params is None:
        _vgg_params = sio.loadmat('/root/bjy/BjyFiles/LavalTrain/Dataset/edgefile/imagenet-vgg-verydeep-19.mat')
    return _vgg_params
def vgg19(input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    weights = vgg_params()['layers'][0]
    net = input_image
    network = {}
    for i,name in enumerate(layers):
        layer_type = name[:4]
        # 若是卷积层
        if layer_type == 'conv':
            kernels,bias = weights[i][0][0][0][0]
            # 由于 imagenet-vgg-verydeep-19.mat 中的参数矩阵和我们定义的长宽位置颠倒了，所以需要交换
            kernels = np.transpose(kernels,(1,0,2,3))
            conv = tf.nn.conv2d(net,tf.constant(kernels),strides=(1,1,1,1),padding='SAME',name=name)
            net = tf.nn.bias_add(conv,bias.reshape(-1))
            net = tf.nn.relu(net)
        # 若是池化层
        elif layer_type == 'pool':
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
        # 将隐藏层加入到集合中
        # 若为`激活函数`直接加入集合
        network[name] = net

    return network
STYLE_WEIGHT = 1
CONTENT_WEIGHT = 1
STYLE_LAYERS = ['relu1_2','relu2_2','relu3_2']
CONTENT_LAYERS = ['pool5']

def loss_function(style_image,content_image,target_image):
    style_features = vgg19([style_image])
    content_features = vgg19([content_image])
    target_features = vgg19([target_image])
    loss = 0.0
    for layer in CONTENT_LAYERS:
        loss += CONTENT_WEIGHT * content_loss(target_features[layer],content_features[layer])

    for layer in STYLE_LAYERS:
        loss += STYLE_WEIGHT * style_loss(target_features[layer],style_features[layer])

    return loss

def content_loss(target_features,content_features):
    bs,height,width,channel = map(lambda i:i.value,content_features.get_shape())
    content_size = height * width * channel * bs
    return tf.nn.l2_loss(target_features - content_features) / content_size#

def style_loss(target_features,style_features):
    _,height,width,channel = map(lambda i:i.value,target_features.get_shape())
    size = height * width * channel
    target_features = tf.reshape(target_features,(-1,channel))
    target_gram = tf.matmul(tf.transpose(target_features),target_features) / size

    style_features = tf.reshape(style_features,(-1,channel))
    style_gram = tf.matmul(tf.transpose(style_features),style_features) / size

    return tf.nn.l2_loss(target_gram - style_gram) / size

def perceptual_loss(gt_hdr, pre_hdr):
    gt_style = vgg19(gt_hdr)
    pre_style = vgg19(pre_hdr)
    print("in vgg19.py gt_style and pre_style", gt_style['pool5'], pre_style['pool5'])
    loss = 0
    for layer in CONTENT_LAYERS:
        loss += CONTENT_WEIGHT * content_loss(gt_style[layer],pre_style[layer])

    return loss