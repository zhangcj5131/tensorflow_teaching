import tensorflow as tf
import argparse
# import cv2
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

#tensorboard --logdir logs --port 6789
class Config:
    def __init__(self):
        self.lr = 0.001
        self.filters = 32
        self.batch_size = 64
        self.classes = 10
        self.eps = 1e-8

        self.epoches = 5
        self.name = 'p08'
        self.save_path = './models/{name}/{name}'.format(name = self.name)
        self.sample_path = './data/MNIST_data'
        self.logdir = './logs/{name}'.format(name = self.name)

    def from_cmd_line(self):
        attrs_dict = self._get_attrs()
        parse = argparse.ArgumentParser()
        for key, value in attrs_dict.items():
            parse.add_argument('--'+key, type = type(value), default=value, help = 'set %s' % key)
        a = parse.parse_args()
        for name in attrs_dict:
            setattr(self, name, getattr(a, name))

    def __repr__(self):
        result = '{'
        for key, value in self._get_attrs().items():
            result += '%s=%s,' % (key, value)
        result = result[0:-1]
        result += '}'
        return result

    def __str__(self):
        return self.__repr__()

class Tensors:
    def __init__(self, config: Config):
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.x = tf.placeholder(tf.float32, [None, 784], 'x')
            x = tf.reshape(self.x, [-1, 28, 28, 1])

            x = tf.layers.conv2d(x, config.filters, 3, 1, 'same', activation = tf.nn.relu, name = 'conv1') #[-1, 28, 28, 32]

            x = tf.layers.conv2d(x, 2*config.filters, 3, 1, 'same', activation = tf.nn.relu, name = 'conv2') #[-1, 28, 28, 64]
            x = tf.layers.max_pooling2d(x, 2, 2) # #[-1, 14, 14, 64]

            x = tf.layers.conv2d(x, 4*config.filters, 3, 1, 'same', activation = tf.nn.relu, name = 'conv3') #[-1, 28, 28, 128]
            x = tf.layers.max_pooling2d(x, 2, 2) # #[-1, 7, 7, 128]

            x = tf.layers.flatten(x)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1000, activation=tf.nn.relu, name = 'dense1')
            y_predict = tf.layers.dense(x, config.classes, name = 'dense2')
            y_predict = tf.nn.softmax(y_predict)
            y_predict = tf.maximum(y_predict, config.eps)

            self.y_digit = tf.argmax(y_predict, axis = 1, output_type=tf.int32)#[1,2,3,5,3---]

            self.y = tf.placeholder(tf.int32, [None], 'y')#[]

            self.precise = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_digit), tf.float32))
            y = tf.one_hot(self.y, config.classes)

            self.loss = -tf.reduce_mean(y*tf.log(y_predict))

            opt = tf.train.AdamOptimizer(self.lr)
            self.train_op = opt.minimize(self.loss)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('precise', self.precise)
            self.summary_op = tf.summary.merge_all()




class Samples:
    def __init__(self, config: Config):
        self.ds = read_data_sets(config.sample_path)

    def next_batch(self, batch_size):
        return self.ds.train.next_batch(batch_size)


    def next_valid(self, batch_size):
        return self.ds.validation.next_batch(batch_size)

    def next_test(self, batch_size):
        return self.ds.test.next_batch(batch_size)

    @property
    def num(self):
        return self.ds.train.num_examples




class Mnist:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        self.samples = Samples(self.config)
        with graph.as_default():
            self.tensors = Tensors(config)
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config = conf)
            self.saver = tf.train.Saver()
            self.file_writer = tf.summary.FileWriter(config.logdir, graph = graph)
            try:
                self.saver.restore(self.session, config.save_path)
                print('resotr the model successfully!')
            except:
                print('the mdoel doese not exst, we have to train a new one')
                self.train()

    def train(self):
        self.session.run(tf.global_variables_initializer())

        batches =self.samples.num // self.config.batch_size
        for epoch in range(self.config.epoches):
            for batch in range(batches):
                x, y = self.samples.next_batch(self.config.batch_size)
                feed_dic = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                    self.tensors.lr: self.config.lr
                }
                _, lo, su = self.session.run([self.tensors.train_op, self.tensors.loss, self.tensors.summary_op], feed_dict=feed_dic)
                self.file_writer.add_summary(su, epoch * batches + batch)

                x, y = self.samples.next_valid(self.config.batch_size)
                feed_dic = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                }
                precise = self.session.run(self.tensors.precise,
                                             feed_dict=feed_dic)

                print('epoch=%d, loss=%f, precise = %f' % (epoch, lo, precise))
        self.saver.save(self.session, self.config.save_path)

    def predict(self):
        x, y = self.samples.next_test(self.config.batch_size)
        feed_dict = {
            self.tensors.x: x,
            self.tensors.y: y
        }
        y_digit, precise = self.session.run([self.tensors.y_digit, self.tensors.precise],
                                   feed_dict=feed_dict)

        for i in range(len(y)):
            print('the predicted value is %d, the true value is %d' % (y_digit[i], y[i]))
        print('the test precise is %f' % (precise))
    def close(self):
        self.session.close()







if __name__ == '__main__':
    config = Config()
    mnist = Mnist(config)
    mnist.predict()
    mnist.close()
