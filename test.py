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

        self.pic_size = 224
        self.filters = 2#64
        self.ch_size = 300

        self.sample_path = 'qts_7X4.txt'

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
    def __init__(self, config: Config, ch_size):
        self.config = config
        self.sub_ts = []

        with tf.device('/gpu:0'):
            self.training = tf.placeholder(tf.bool, [], 'training')
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('poem'):
            for gpu_index in range(config.gpus):
                self.sub_ts.append(SubTensors(config, gpu_index, opt, ch_size, self.training))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grad = self.merge_grads()

            self.train_op = opt.apply_gradients(grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_ts])

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            self.precise = tf.reduce_mean([ts.precise for ts in self.sub_ts])

    def merge_grads(self):
        indexed_grads = {}
        grads = {}
        for ts in self.sub_ts:
            for g, v in ts.grad:
                if isinstance(g, tf.IndexedSlices):
                    if not v in indexed_grads:
                        indexed_grads[v] = []
                    indexed_grads[v].append(g)
                else:
                    if not v in grads:
                        grads[v] = []
                    grads[v].append(g)
        # grads = { v1: [g11, g12, g13, ...], v2:[g21, g22, ...], ....}
        grads = [(tf.reduce_mean(grads[v], axis=0), v) for v in grads]

        for v in indexed_grads:
            indices = tf.concat([g.indices for g in indexed_grads[v]], axis=0)
            values = tf.concat([g.values for g in indexed_grads[v]], axis=0)
            g = tf.IndexedSlices(values, indices)
            grads.append((g, v))

        return grads


class SubTensors:
    def __init__(self, config, gpu_index, opt, ch_size, training):
        self.config = config
        self.training = training
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.float32, [None, config.pic_size, config.pic_size, 3], 'x')

            with tf.variable_scope('resnet'):
                x = self.resnet50(self.x)  # [-1, num_units]

            cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name='lstm')
            # peephole_cell = tf.nn.rnn_cell.LSTMCell(use_peepholes=True)

            # cell = MyLSTMCell(config.num_units, 'lstm')
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
            # cell = MyMultiRNNCell([cell] * 2)

            batch_size_ts = tf.shape(self.x)[0]
            state = cell.zero_state(batch_size_ts, tf.float32)  # [batch_size, state_size]

            # state = [tf.nn.rnn_cell.LSTMStateTuple, tf.nn.rnn_cell.LSTMStateTuple]

            self.y = tf.placeholder(tf.int32, [None, config.num_step], 'y')
            y = tf.one_hot(self.y, ch_size)  # [-1, num_step, ch_size]

            losses = []
            precises = []
            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    yi_pred, state = cell(x, state)  # [-1, num_units]

                    yi_pred = tf.layers.dense(yi_pred, ch_size, name='yi_dense')  # [-1, ch_size]
                    yi = y[:, i, :]  # [-1, ch_size]

                    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yi, logits=yi_pred)
                    losses.append(loss_i)
                    tf.get_variable_scope().reuse_variables()

                    yi_pred0 = tf.argmax(yi_pred, axis=1, output_type=tf.int32)  # [-1]
                    yi0 = self.y[:, i]  # [-1]
                    precise = tf.equal(yi_pred0, yi0)
                    precise = tf.cast(precise, tf.float32)
                    precise = tf.reduce_mean(precise)
                    precises.append(precise)

            self.loss = tf.reduce_mean(losses)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.grad = opt.compute_gradients(self.loss)  # [(g1, v1), (g2, v2), ... , (gn, vn)]
            self.precise = tf.reduce_mean(precises)

    def resnet50(self, x):
        cfg = self.config
        x = tf.layers.conv2d(x, cfg.filters, 7, 2, 'same', name='conv1')
        x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=self.training, name='bn1')
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 3, 2, 'same')

        filters = cfg.filters
        for i, module in enumerate([3, 4, 6, 3]):
            for j in range(module):
                with tf.variable_scope('resnet_%d_%d' % (i, j)):
                    strides = 2 if i > 0 and j == 0 else 1
                    with tf.variable_scope('left'):
                        left = self.resnet_left(x, filters, strides)
                    with tf.variable_scope('right'):
                        right = self.resnet_right(x, filters, strides)
                x = tf.nn.relu(left + right)
            filters *= 2
        x = tf.layers.average_pooling2d(x, 7, 7)  # [-1, 1, 1, 2048]
        x = tf.layers.flatten(x)  # tf.reshape(x, [-1, 2048])
        x = tf.layers.dense(x, cfg.num_units, name='dense')
        return x

    def resnet_left(self, x, filters, strides):
        x = tf.layers.conv2d(x, filters, 1, strides, 'same', name='conv1')
        x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=self.training, name='bn1')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters, 3, 1, 'same', name='conv2')
        x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=self.training, name='bn2')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, 4*filters, 1, 1, 'same', name='conv3')
        x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=self.training, name='bn3')
        return x

    def resnet_right(self, x, filters, strides):
        if strides > 1 or 4 * filters != x.shape[-1].value:
            x = tf.layers.conv2d(x, 4 * filters, 1, strides, 'same', name='conv')
            x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=self.training, name='bn')  # should call control-dependencies() explicitly
        return x


class Samples:
    def __init__(self, config: Config):
        self.config = config

    def num(self):
        return 1000

    def next_batch(self, batch_size):
        x = np.random.uniform(size=[batch_size, self.config.pic_size, self.config.pic_size, 3])
        y = np.random.randint(0, self.config.ch_size, size=[batch_size, self.config.num_step])
        return x, y


class PicTitle:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.samples = Samples(config)
            self.tensors = Tensors(config, config.ch_size)
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, config.save_path)
                print('Restore the model from %s successfully.' % config.save_path)
            except:
                print('Fail to restore the model from', config.save_path)
                self.session.run(tf.global_variables_initializer())

    def train(self):
        cfg = self.config
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)

        for epoch in range(cfg.epoches):
            batches = self.samples.num() // (cfg.gpus * cfg.batch_size)
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: cfg.lr,
                    self.tensors.training: True
                }
                for gpu_index in range(cfg.gpus):
                    x, y = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_ts[gpu_index].x] = x
                    feed_dict[self.tensors.sub_ts[gpu_index].y] = y
                _, loss, su, precise = self.session.run(
                    [self.tensors.train_op, self.tensors.loss, self.tensors.summary_op, self.tensors.precise], feed_dict)
                writer.add_summary(su, epoch * batches + batch)
                print('%d/%d: loss=%.8f, precise=%.8f' % (batch, epoch, loss, precise), flush=True)
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path, flush=True)

    def predict(self, head=None):
        pass

    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()

    app = PicTitle(config)

    app.train()
    app.close()
    print('Finished!')
