import tensorflow as tf
import numpy as np
import cv2
import argparse
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

class Config:
    def __init__(self):
        self.batch_size = 400
        self.epoches = 3
        self.eps = 1e-8
        self.name = 'p14'
        self.lr = 0.0002
        self.save_path = 'models/{name}/{name}'.format(name=self.name)
        self.sample_path = './data/MNIST_data'
        self.filters = 32
        self.logdir = './logs/{name}'.format(name=self.name)
        self.vector_size = 1
        self.decay_rate = 0.995
        self.image_path = 'images/{name}/%s.jpg'.format(name=self.name)


    def from_cmd_line(self):
        attrs = self._get_attrs()
        parser = argparse.ArgumentParser()
        for name, value in attrs.items():
            print('get %s from cmd' % name)
            parser.add_argument('--'+name, type = type(value), default=value, help='default=%s' % value)
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
            self.x = tf.placeholder(tf.float32, [None, 28*28], 'x')
            x = tf.reshape(self.x, [-1, 28, 28, 1])
            with tf.variable_scope('encode'):
                vector = self.encode(x)
            vector = self.normalize(vector)

            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.y = tf.placeholder(tf.int32, [None], 'y')
            y = tf.one_hot(self.y, 10)
            with tf.variable_scope('decode'):
                x2 = self.decode(vector, y)
                self.x2 = tf.reshape(x2, [-1, 28, 28]) * 255

            loss = tf.reduce_mean(tf.square(x2 - x))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            self.loss = tf.sqrt(loss)

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def encode(self, x): # [-1, 28, 28, 1]
        cfg = self.config
        x = tf.layers.conv2d(x, cfg.filters, 3, 1, 'same', activation = tf.nn.relu, name = 'conv1')#[-1, 28, 28, 32]
        x = tf.layers.conv2d(x, 2*cfg.filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2')  # [-1, 14, 14, 64]
        x = tf.layers.conv2d(x, 4 * cfg.filters, 3, 2, 'same', activation=tf.nn.relu, name='conv3')  # [-1, 7, 7, 128]

        x = tf.layers.flatten(x) # [-1, 7*7*128]
        x = tf.layers.dense(x, 1000, activation = tf.nn.relu, name = 'dense1')
        vector = tf.layers.dense(x, cfg.vector_size, name='dense2')
        return vector

    def normalize(self,vector):
        cfg = self.config
        mom_mean = tf.get_variable('mom_mean', [cfg.vector_size], tf.float32,
                                   initializer=tf.initializers.zeros, trainable=False)
        mom_square = tf.get_variable('mom_squre', [cfg.vector_size], tf.float32,
                                   initializer=tf.initializers.zeros, trainable=False)

        mean = tf.reduce_mean(vector, axis = 0)
        new_mean = (1 - cfg.decay_rate) * mom_mean + cfg.decay_rate * mean

        square = tf.reduce_mean(tf.square(vector), axis = 0)
        new_square = (1 - cfg.decay_rate) * mom_square + cfg.decay_rate * square

        #mom_mean = new_mean
        mean_assign = tf.assign(mom_mean, new_mean)
        square_assign = tf.assign(mom_square, new_square)

        with tf.control_dependencies([mean_assign, square_assign]):
            #s*s = E(x**2) - (E(x)*E(x))
            std = tf.sqrt(mom_square - tf.square(mom_mean))
            std = tf.maximum(std, cfg.eps)
            self.vector = (vector - mom_mean) / std

        std = tf.sqrt(mom_square - tf.square(mom_mean))
        std = tf.maximum(std, cfg.eps)
        return self.vector*std + mom_mean











    def decode(self, vector, y):
        vector = tf.layers.dense(vector, 1000, name = 'dense1') #[-1, 10000]
        yy = tf.layers.dense(y, 1000, name = 'y_dense1')
        vector += yy
        vector = tf.nn.relu(vector)

        vector = tf.layers.dense(vector, 7*7*128, name = 'dense2')#[-1, 7*7*128]
        yy = tf.layers.dense(y, 7*7*128, name = 'y_dense2')
        vector += yy
        vector = tf.nn.relu(vector)

        cfg = self.config
        x2 = tf.reshape(vector, [-1, 7, 7, 128])

        x2 = tf.layers.conv2d_transpose(x2, 2 * cfg.filters, 3, 2, 'same', name = 'deconv1') #[-1, 14, 14, 64]
        yy = tf.layers.dense(y, 14*14*64, name = 'y_desne3') #[-1, 14*14*64]
        yy = tf.reshape(yy, [-1, 14, 14, 64])
        x2 += yy
        x2 = tf.nn.relu(x2)

        x2 = tf.layers.conv2d_transpose(x2, cfg.filters, 3, 2, 'same', name = 'deconv2') #[-1, 28, 28, 32]
        yy = tf.layers.dense(y, 28*28*32, name = 'y_desne4') #[-1, 28, 28, 32]
        yy = tf.reshape(yy, [-1, 28, 28, 32])
        x2 += yy
        x2 = tf.nn.relu(x2)

        x2 = tf.layers.conv2d_transpose(x2, 1, 3, 1, 'same', name = 'deconv3') #[-1, 28, 28, 1]
        yy = tf.layers.dense(y, 28*28*1, name = 'y_desne5') #[-1, 28, 28, 32]
        yy = tf.reshape(yy, [-1, 28, 28, 1])
        x2 += yy
        x2 = tf.nn.relu(x2)

        return x2





class Samples:
    def __init__(self, config: Config):
        self.ds = read_data_sets(config.sample_path)

    def next_batch(self, batch_szie):
        return self.ds.train.next_batch(batch_szie)

    @property
    def num(self):
        return self.ds.train.num_examples

class Mnist:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config = conf, graph = graph)
            self.tensors = Tensors(config)
            self.saver = tf.train.Saver()
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph = graph)
            try:
                self.saver.restore(self.session, config.save_path)
                print('restore the model successfully')
            except:
                print('the mdoel does not exist, we have to train a new one!')
                self.train()

    def train(self):
        self.session.run(tf.global_variables_initializer())

        cfg = self.config
        self.samples = Samples(cfg)
        for epoch in range(cfg.epoches):
            batches = self.samples.num // cfg.batch_size
            for batch in range(batches):
                x, y = self.samples.next_batch(cfg.batch_size)
                feed_dict = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                    self.tensors.lr: cfg.lr
                }
                _, lo, su = self.session.run([self.tensors.train_op,
                                  self.tensors.loss,
                                  self.tensors.summary_op], feed_dict=feed_dict)
                print('%d/%d, loss = %f' % (epoch, batch,lo))
                self.file_writer.add_summary(su, epoch*batches + batch)
            self.saver.save(self.session, cfg.save_path)
            self.predict(epoch)

    def predict(self, epoch):
        cfg =  self.config
        vector = np.random.normal(size = [cfg.batch_size, cfg.vector_size])
        y = [e % 10 for e in range(cfg.batch_size)]
        feed_dict = {
            self.tensors.vector: vector,
            self.tensors.y: y
        }
        x2 = self.session.run(self.tensors.x2, feed_dict)#[-1,28,28]
        x2 = np.transpose(x2, [1, 0, 2])
        x2 = np.reshape(x2, [28, -1, 28*20])
        x2 = np.transpose(x2, [1,0,2])
        x2 = np.reshape(x2, [-1, 28*20])
        x2 = np.uint8(x2)

        # cv2.imshow('mypic', x2)
        # cv2.waitKey()

        path = cfg.image_path % epoch
        cv2.imwrite(path, x2)
        print('image saved to ', path, flush = True)

    def close(self):
        self.session.close()

if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()

    mnist = Mnist(config)

    mnist.close()

