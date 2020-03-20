import tensorflow as tf
import numpy as np
import argparse
import os


class Config:
    def __init__(self):
        self.batch_size = 2
        self.num_step = 8 * 4
        self.num_units = 10
        self.gpus = self.get_gpus()
        self.filters = 2#64
        self.picture_size = 224

        self.name = 'p24'
        self.save_path = 'models/{name}/{name}'.format(name=self.name)
        self.logdir = 'logs/{name}/'.format(name=self.name)

        self.lr = 0.0002
        self.epoches = 100

    def get_gpus(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        return len(value.split(','))

    def from_cmd_line(self):
        parser = argparse.ArgumentParser()
        for name in dir(self):
            value = getattr(self, name)
            if type(value) in (int, float, bool, str) and not name.startswith('__'):
                parser.add_argument('--' + name, default=value, help='Default to %s' % value, type=type(value))
        a = parser.parse_args()
        for name in dir(self):
            value = getattr(self, name)
            if type(value) in (int, float, bool, str) and not name.startswith('__') and hasattr(a, name):
                value = getattr(a, name)
                setattr(self, name, value)

class Tensors:
    def __init__(self, config: Config, char_size):
        self.config = config
        self.sub_tensors = []
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.training = tf.placeholder(tf.bool, [], 'training')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('picture_title'):
            for gpu_index in range(config.gpus):
                self.sub_tensors.append(Sub_tensors(config, gpu_index, opt, char_size, self.training))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grad = self.merge_grads()
            self.train_op = opt.apply_gradients(grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_tensors])
            self.precise = tf.reduce_mean([ts.precise for ts in self.sub_tensors])

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('precise', self.precise)
            self.summary_op = tf.summary.merge_all()

    def merge_grads(self):
        grads = {}
        indexed_grads = {}
        for ts in self.sub_tensors:
            for g, v in ts.grad:
                if isinstance(g, tf.IndexedSlices):
                    if v not in indexed_grads:
                        indexed_grads[v] = []
                    indexed_grads[v].append(g)
            else:
                if v not in grads:
                    grads[v] = []
                grads[v].append(g)
        result = [(tf.reduce_mean(grads[v], axis = 0), v) for v in grads]
        for v in indexed_grads:
            indices = tf.concat([g.indices for g in indexed_grads[v]], axis = 0)
            values = tf.concat([g.values for g in indexed_grads[v]], axis=0)
            g = tf.IndexedSlices(values, indices)
            result.append((g, v))
        return result


class Sub_tensors:
    def __init__(self, config: Config, gpu_index, opt: tf.train.AdamOptimizer, char_size, training):
        self.config = config
        self.training = training
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.float32, [None, config.picture_size, config.picture_size, 3], 'x')
            with tf.variable_scope('resnet50'):
                x = self.resnet50(self.x)

            lstm = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name = 'lstm')
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2)

            batch_size_tf = tf.shape(self.x)[0]
            state = cell.zero_state(batch_size_tf, tf.float32)

            self.y = tf.placeholder(tf.int32, [None, config.num_step], 'y')
            y_onehot = tf.one_hot(self.y, char_size)#-1, num_step, char_size

            loss_list = []
            precise_list = []

            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    y_pred, state = cell(x, state)
                    y_pred = tf.layers.dense(y_pred, char_size, name = 'y_predict')
                    tf.get_variable_scope().reuse_variables()

                    y = y_onehot[:,i]
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y)
                    loss_list.append(loss)

                    y_predict_id = tf.math.argmax(y_pred, axis = 1, output_type=tf.int32)
                    y_id = self.y[:,i]
                    precise = tf.equal(y_predict_id, y_id)
                    precise = tf.cast(precise, tf.float32)
                    precise = tf.reduce_mean(precise)
                    precise_list.append(precise)

            self.loss = tf.reduce_mean(loss_list)
            self.precise = tf.reduce_mean(precise_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.grad = opt.compute_gradients(self.loss)


    def resnet50(self, x):
        cfg = self.config
        x = tf.layers.conv2d(x, cfg.filters, 7, 2, 'same', name = 'conv1')
        x = tf.layers.batch_normalization(x, [1,2,3], name = 'bn1')
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 3, 2, 'same')

        filters = cfg.filters

        for i, model in enumerate([3,4,6,3]):
            for j in range(model):
                strides = 2 if i > 0 and j == 0 else 1
                with tf.variable_scope('resnet_%d_%d' % (i, j)):
                    with tf.variable_scope('left'):
                        left = self.left_net(x, strides, filters)
                    with tf.variable_scope('right'):
                        right = self.right_net(x, strides, filters)
                x = tf.nn.relu(left + right)
            filters *= 2
        x = tf.layers.max_pooling2d(x, 7, 7)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, cfg.num_units, name = 'dense')
        return x

    def left_net(self, x, strides, filters):
        x = tf.layers.conv2d(x, filters, 1, strides, 'same', name = 'conv1')
        x = tf.layers.batch_normalization(x, [1,2,3], name = 'bn1')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters, 3, 1, 'same', name = 'conv2')
        x = tf.layers.batch_normalization(x, [1,2,3], name = 'bn2')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, 4*filters, 1, 1, 'same', name = 'conv3')
        x = tf.layers.batch_normalization(x, [1,2,3], name = 'bn3')
        return x

    def right_net(self, x, strides, filters):
        if strides == 2 or tf.shape(x)[-1] != 4*filters:
            x = tf.layers.conv2d(x, 4*filters, 1, strides, 'same', name = 'con')
            x = tf.layers.batch_normalization(x, [1,2,3], name = 'bn')
        return x





class Samples:
    def __init__(self, config: Config):
        self.config = config

    @property
    def char_size(self):
        return 300

    @property
    def num(self):
        return 100

    def next_batch(self, batch_size):
        x = np.random.uniform(size = [batch_size, self.config.picture_size, self.config.picture_size, 3])
        y = np.random.randint(0, self.char_size, size = [batch_size, self.config.num_step])
        return x, y




class Pic_title:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config=conf, graph=graph)
            self.samples = Samples(config)
            self.tensors = Tensors(config, self.samples.char_size)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph=graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, config.save_path)
            except:
                self.session.run(tf.global_variables_initializer())

    def train(self):
        config = self.config
        batches = self.samples.num // (config.gpus * config.batch_size)
        for epoch in range(config.epoches):
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: config.lr,
                    self.tensors.training: True
                }
                for gpu_index in range(config.gpus):
                    x, y = self.samples.next_batch(config.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = x
                    feed_dict[self.tensors.sub_tensors[gpu_index].y] = y
                _, loss, precise, su = self.session.run([self.tensors.train_op,
                                  self.tensors.loss,
                                  self.tensors.precise,
                                  self.tensors.summary_op], feed_dict=feed_dict)
                self.file_writer.add_summary(su, epoch*batches + batch)
                print('%d/%d, loss=%f, precise=%f' % (epoch, batch, loss, precise))
            self.saver.save(self.session, config.save_path)




    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    app = Pic_title(config)
    app.train()

