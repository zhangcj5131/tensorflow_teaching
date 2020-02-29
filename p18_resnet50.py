import tensorflow as tf
import numpy as np
import argparse

import os



class Config:
    def __init__(self):
        self.batch_size = 2
        self.epoches = 10
        self.eps = 1e-8
        self.lr = 0.001
        self.name = 'p18'
        self.save_path = 'models/{name}/{name}'.format(name=self.name)
        self.sample_path = '/Users/cjz/data/pic'
        self.filters = 2
        self.logdir = './logs/{name}'.format(name=self.name)
        self.classes = 10  # should be about 90000 in practice.

    def from_cmd_line(self):
        attrs = self._get_attrs()
        parser = argparse.ArgumentParser()
        for name, value in attrs.items():
            print('add %s from cmd' % name)
            parser.add_argument('--'+name, type=type(value), default=value, help='default=%s' % value)
        a = parser.parse_args()
        for name in attrs:
            setattr(self, name, getattr(a, name))

    def _get_attrs(self):
        attrs = {}
        for name in dir(self):
            value = getattr(self, name)
            if type(value) in (int, float, str, bool) and not name.startswith('__'):
                attrs[name] = value
        return attrs



    def __repr__(self):  # representation
        """
        Called by using operator % between a string and this object
        :return:
        """
        result = '{'
        attrs = self._get_attrs()
        for name in attrs:
            result += ' %s = %s,' % (name, attrs[name])
        return result + '}'

    def __str__(self):
        """
        Called by str(object)
        :return: the string of this object
        """
        return self.__repr__()


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        with tf.device('/gpu:0'):
            self.training = tf.placeholder(tf.bool, [], 'training')
            self.x = tf.placeholder(tf.float32, [None, 224, 224, 3], 'x')
            x = self.x / 255
            with tf.variable_scope('resnet'):
                logit = self.resnet(x)

            y_predict = tf.nn.softmax(logit)
            self.y_predict = tf.math.argmax(y_predict, axis = 1, output_type=tf.int32)

            self.y = tf.placeholder(tf.int32, [None], 'y')
            y = tf.one_hot(self.y, config.classes)

            y_predict = tf.maximum(y_predict, config.eps)
            self.loss = -tf.reduce_mean(y*tf.log(y_predict))
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = opt.minimize(self.loss)

            precision = tf.cast(tf.equal(self.y, self.y_predict), tf.float32)
            self.precision = tf.reduce_mean(precision)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('precision', self.precision)
            self.summary_op = tf.summary.merge_all()

    def resnet(self, x):
        cfg = self.config
        filters = cfg.filters
        x = tf.layers.conv2d(x, filters, 7, 2, 'same', name = 'conv1')
        x = tf.layers.batch_normalization(x, axis = [1,2,3], training = self.training, name = 'bn1')
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 3, 2, 'same')

        for i, model in enumerate([3,4,6,3]):
            for j in range(model):
                strides = 2 if i != 0 and j == 0 else 1
                with tf.variable_scope('resnet_%s_%s' % (i, j)):
                    with tf.variable_scope('left'):
                        left = self.resnet_left(x, filters, strides)
                    with tf.variable_scope('right'):
                        right = self.resnet_right(x, filters, strides)
                    x = tf.nn.relu(left + right)
            filters *= 2
        x = tf.layers.average_pooling2d(x, 7, 7)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, cfg.classes, name = 'dense')
        return x

    def resnet_left(self, x, filters, strides):
        x = tf.layers.conv2d(x, filters, 1, strides, 'same', name = 'conv1')
        x = tf.layers.batch_normalization(x, axis = [1,2,3], training = self.training, name = 'bn1')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters, 3, 1, 'same', name = 'conv2')
        x = tf.layers.batch_normalization(x, axis = [1,2,3], training = self.training, name = 'bn2')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, 4*filters, 1, 1, 'same', name = 'conv3')
        x = tf.layers.batch_normalization(x, axis = [1,2,3], training = self.training, name = 'bn3')
        return x

    def resnet_right(self, x, filters, strides):
        if strides != 1 or x.shape[-1].value != 4*filters:
            x = tf.layers.conv2d(x, 4*filters, 1, strides, 'same', name = 'conv')
            x = tf.layers.batch_normalization(x, axis = [1,2,3], training = self.training, name = 'bn')
        return x





def my_batch_normalization(x, momentum=0.99,
                        epsilon=1e-3, training=False,):
    shape = [ch.value for ch in x.shape]
    shape = shape[1:]

    mom_mean = tf.get_variable('mom_mean', shape, tf.float32, tf.initializers.zeros, trainable=False)
    mom_square = tf.get_variable('mom_square', shape, tf.float32, tf.initializers.zeros, trainable=False)

    def training_true():
        mean = tf.reduce_mean(x, axis = 0)#[1,2,3]
        new_mean = momentum*mom_mean + (1-momentum)*mean
        assign_mean = tf.assign(mom_mean, new_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_mean)

        square = tf.reduce_mean(x**2, axis=0)
        new_square = momentum*mom_square + (1-momentum)*square
        assign_square = tf.assign(mom_square, new_square)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_square)
        return mom_mean, mom_square

    def training_false():
        return mom_mean, mom_square

    mean, square = tf.cond(training, training_true, training_false)

    std = tf.sqrt(square - mean**2)
    std = tf.maximum(std, epsilon)
    x = (x - mean)/std

    beta = tf.get_variable('beta', shape, tf.float32, tf.initializers.zeros, trainable=True)
    gamma = tf.get_variable('gamma', shape, tf.float32, tf.initializers.zeros, trainable=True)
    return x*beta + gamma







class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.x = np.random.uniform(low=0, high=255, size=[self.num//config.batch_size, config.batch_size, 224, 224, 3])
        self.y = np.random.randint(0, config.classes, size = [self.num//config.batch_size, config.batch_size])
        self.index = 0

    def next_batch(self, batch_size):
        x = self.x[self.index, :, :, :, :]#[batch_szie, 224, 224, 3]
        y = self.y[self.index, :]
        self.index = (self.index + 1) % np.shape(self.x)[0]
        return x, y

    @property
    def num(self):
        return 100

    def close(self):
        pass




class Face_recognition:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.samples = Samples(config)
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config = conf, graph = graph)
            self.tensors = Tensors(config)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph = graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, config.save_path)
                print('restore the model successfully!')
            except:
                print('the model does not exist, we have to train a new one')
                self.train()

    def train(self):
        cfg = self.config

        self.session.run(tf.global_variables_initializer())
        for epoch in range(cfg.epoches):
            batches = self.samples.num // cfg.batch_size
            for batch in range(batches):
                x, y = self.samples.next_batch(cfg.batch_size)
                feed_dict = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                    self.tensors.lr: cfg.lr,
                    self.tensors.training: True
                }
                _, lo, su = self.session.run([self.tensors.train_op,
                                  self.tensors.loss,
                                  self.tensors.summary_op], feed_dict)
                print('%d/%d, loss = %f' % (epoch, batch, lo))
                self.file_writer.add_summary(su, epoch*batches + batch)
            self.saver.save(self.session, cfg.save_path)

    def close(self):
        self.session.close()
        self.samples.close()

if __name__ == '__main__':
    print('CUDA_VISIBLE_DEVICES = %s' % os.getenv('CUDA_VISIBLE_DEVICES', '0'))
    config = Config()
    config.from_cmd_line()

    face = None
    try:
        face = Face_recognition(config)
    finally:
        face.close()
    # tensor = Tensors(config)
    # print(config)
















