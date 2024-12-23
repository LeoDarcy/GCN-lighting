from gcn.layers import *
from gcn.render_tf import *
from gcn.vgg19 import *
import tensorflow as tf
#from gcn.utils_np import render_loss
flags = tf.app.flags
FLAGS = flags.FLAGS


class BjyModel(object):
    def __init__(self, input_name, batchsize, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        self.name = input_name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.CNN_layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.paramloss = tf.cast(0, tf.float32)
        self.renderloss = tf.cast(0, tf.float32)
        self.perceploss = tf.cast(0, tf.float32)
        self.edge_loss = tf.cast(0, tf.float32)
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.batch_size = batchsize

        self.g_optimizer = None
        self.g_opt_op = None
        self.g_loss = 0
        self.d_optimizer = None
        self.d_opt_op = None
        self.d_loss = 0
    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        #tf.print(self.inputs)
        for layer in self.CNN_layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.CNN_output = self.activations[-1]
        #print(self.CNN_output)
        #Global average pooling
        #self.CNN_output = tf.reduce_mean(self.CNN_output,[1, 2])
        #转换为feature来执行
        reshape_feature = tf.reshape(self.CNN_output, [self.batch_size, 128, self.input_dim])
        self.activations.append(reshape_feature)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = tf.reshape(self.activations[-1], [self.batch_size, 128, 3])

        # Store model variables for easy access
        
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print(variables)
        #assert(1==0)
        #variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.vars = variables
        self.printVars = {var.name: var for var in variables}
        print(self.vars, len(self.vars))
        #assert(1==0)

        # Build metrics
        self._loss()
        self._accuracy()

        self._gloss()
        self.opt_op = self.optimizer.minimize(self.loss, var_list=self.vars)
        self.g_opt_op = self.optimizer.minimize(self.g_loss)
        #self.d_opt_op = self.optimizer.minimize(self.d_loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, info, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(var_list=self.vars, max_to_keep=1)
        #print(self.vars)
        print("in save model !!!!!   ", len(self.vars))
        #result = sess.run(self.vars)
        #np.save("./train_param.npy", result)
        #print(result)
        
        save_path = saver.save(sess, "tmp/%s_test.ckpt" % (self.name))
        print("Model saved in file: %s" % save_path)
        #assert(1==0)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(var_list=self.vars, max_to_keep=1)
        save_path = "tmp/%s_test.ckpt" % self.name
        #print(save_path)
        saver.restore(sess, save_path)
        
        #result = sess.run(self.vars)
        #np.save("./test_param.npy", result)
        #print(self.vars)
        #print(len(self.vars))
        #assert(1==0)
        print("Model restored from file: %s" % save_path)

    def Print(self):
        print(self.printVars)
    def _gloss(self):
        raise NotImplementedError

class BjyGCN(BjyModel):
    def __init__(self, placeholders, input_name, input_dim, **kwargs):
        #input_dim = position + feature + color :3 + 23*15 + 3 = 351
        super(BjyGCN, self).__init__(input_name, **kwargs)

        self.inputs = placeholders['img_input']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[2]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        a = tf.reshape(self.outputs, [self.batch_size, 1, -1])
        b = tf.tile(a, (1, 128, 1))
        c = tf.tile(self.outputs, (1, 1,128))
        sub = tf.abs(b - c)
        print(a, b, c)
        sub = tf.reshape(sub, (self.batch_size, -1, 3))
        print("sub", sub)
        sub = tf.reduce_sum(sub, axis = 2)
        print("sub", sub)
        sub = tf.reshape(sub, (self.batch_size, -1, 128))
        print("sub", sub)
        mul = tf.matmul(sub, self.placeholders['adj_array'])
        print("mul", mul)
        diag = tf.map_fn(tf.diag_part, mul)
        print("mul", diag)

        a = tf.reshape(self.placeholders['labels'], [self.batch_size, 1, -1])
        b = tf.tile(a, (1, 128, 1))
        c = tf.tile(self.placeholders['labels'], (1, 1,128))
        sub = tf.abs(b - c)
        sub = tf.reshape(sub, (self.batch_size, -1, 3))
        sub = tf.reduce_sum(sub, axis = 2)
        sub = tf.reshape(sub, (self.batch_size, -1, 128))
        mul = tf.matmul(sub, self.placeholders['adj_array'])
        diag_label = tf.map_fn(tf.diag_part, mul)
        
        self.edge_loss = tf.reduce_mean(tf.abs(diag - diag_label))

        image_pre = render_loss_tensorflow(self.outputs)
        image_gt = render_loss_tensorflow(self.placeholders['labels'])
        #image_gt = tf.reshape(self.placeholders['origin_hdr'], (-1,3))
        
        render_loss = image_pre - image_gt
        render_loss = (render_loss ** 2) * 1.0

        #perceptual loss
        #image_gt = self.placeholders['origin_hdr']
        image_pre = tf.reshape(image_pre, (self.batch_size,64, 128, 3))
        image_gt = tf.reshape(image_gt, (self.batch_size,64, 128, 3))
        
        percep_loss = perceptual_loss(image_gt, image_pre)
        self.perceploss = tf.reduce_mean(percep_loss)

        sub_result = self.outputs - self.placeholders['labels']
        sub_result = (sub_result ** 2) * 1.0
        self.paramloss = tf.reduce_mean (sub_result * self.placeholders['label_mask'])
        
        self.renderloss = tf.reduce_mean(render_loss)
        self.loss = self.paramloss + 0.2 * self.renderloss + 0.1 * self.perceploss#+ 0.01 * self.edge_loss
        return self.loss
        
    def _gloss(self):
        self.g_loss = self.loss
    
    def _accuracy(self):
        self.accuracy = 0
        #self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                self.placeholders['labels_mask'])

    def _build(self):
        with tf.variable_scope(self.name):
            self.CNN_layers.append(tf.layers.Conv2D(filters = 32, kernel_size = 7, padding = 'same', activation=tf.nn.relu))
            self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
            self.CNN_layers.append(tf.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
            self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
            self.CNN_layers.append(tf.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
            self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
            self.CNN_layers.append(tf.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
            self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
            self.CNN_layers.append(tf.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
            self.CNN_layers.append(tf.layers.MaxPooling2D(pool_size = 2, strides = 2))
            #self.CNN_layers.append(tf.layers.Conv2D(filters = 128*3*5, kernel_size = 3, padding = 'same', activation=tf.nn.relu))
            
            self.CNN_layers.append(tf.layers.Flatten())
            self.CNN_layers.append(tf.layers.Dense(units = 128 * 3 * 10, activation=tf.nn.relu))
            self.CNN_layers.append(tf.layers.Dense(units = 128*self.input_dim, activation=tf.nn.relu))
            
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden,
                                                placeholders=self.placeholders,
                                                index = 0,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                sparse_inputs=False,
                                                logging=self.logging))
            
            for _ in range(4):
                self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                placeholders=self.placeholders,
                                                index = 0,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                sparse_inputs=False,
                                                logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                index = 0,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                logging=self.logging))
            #self.layers.append(tf.layers.Dense(units=3))
        
    def predict(self):
        return self.outputs * 100

