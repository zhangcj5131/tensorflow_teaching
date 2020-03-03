import tensorflow as tf
import numpy as np
import argparse
import os


class Config:
    def __init__(self):
        #多少只股票
        self.batch_size = 10
        #用多少天的数据预测
        self.num_step = 8
        self.state_size = 2
        self.hidden_size = 7
        self.gpus = self.get_gpus()

        self.days = 100
        self.name = 'p21'
        self.save_path = 'models/{name}/{name}'.format(name=self.name)
        self.logdir = 'logs/{name}/'.format(name=self.name)

        self.lr = 0.0002
        self.epoches = 20

    def get_gpus(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '2,4')
        return len(value.split(','))

    def from_cmd_line(self):
        parser = argparse.ArgumentParser()
        for name in dir(self):
            value = getattr(self, name)
            if type(value) in (int, float, bool, str) and not name.startswith('__'):
                parser.add_argument('--' + name, default=value, help='Default to %s' % value)
        a = parser.parse_args()
        for name in dir(self):
            value = getattr(self, name)
            if type(value) in (int, float, bool, str) and not name.startswith('__') and hasattr(a, name):
                value = getattr(a, name)
                setattr(self, name, value)


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []

        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('stock'):
            for gpu_index in range(config.gpus):
                self.sub_ts.append(SubTensors(config, gpu_index, opt))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grad = self.merge_grads()

            self.train_op = opt.apply_gradients(grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_ts])

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def merge_grads(self):
        grads = {}
        for ts in self.sub_ts:
            for g, v in ts.grad:
                if not v in grads:
                    grads[v] = []
                grads[v].append(g)

        # for k, v in grads.items():
        #     print(len(v))

        grads = [(tf.reduce_mean(grads[v], axis=0), v) for v in grads]
        # print(len(grads))
        return grads


class SubTensors:
    def __init__(self, config, gpu_index, opt):
        with tf.device('/gpu:%d' % gpu_index):

            self.x = tf.placeholder(tf.float32, [config.num_step, None], 'x')
            self.y = tf.placeholder(tf.float32, [None], 'y')

            cell = MyCell(config.state_size, config.hidden_size)
            state = cell.zero_state(tf.shape(self.x)[1])  # [batch_size, state_size]

            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    xi = self.x[i, :]
                    state = cell(xi, state, name='my_cell')
                    tf.get_variable_scope().reuse_variables()

            y_predict = tf.layers.dense(state, 1, name='dense')  # [-1, 1]
            self.y_predict = tf.reshape(y_predict, [-1])

            self.loss = tf.reduce_mean(tf.abs(self.y_predict - self.y))
            #(g1, v1),v是变量名,g 是变量值
            self.grad = opt.compute_gradients(self.loss)  # [(g1, v1), (g2, v2), ... , (gn, vn)]

            # if gpu_index == 0:
            #     for g, v in self.grad:
            #         print(g)



class MyCell:
    def __init__(self, state_size, hidden_size):
        self.state_size = state_size
        self.hidden_size = hidden_size

    def zero_state(self, batch_size, dtype=tf.float32):
        return tf.zeros([batch_size, self.state_size], dtype)

    def __call__(self, xi, state, name):
        """

        :param xi: [-1]
        :param state: [batch_size, state_size]
        :param name:
        :return: [-1, 2]
        """
        with tf.variable_scope(name):
            xi = tf.reshape(xi, [-1, 1])
            xi = tf.concat((xi, state), axis=1)  # [-1, 3]
            xi = tf.layers.dense(xi, self.hidden_size, tf.nn.relu, name='dense1')
            state = tf.layers.dense(xi, self.state_size, name='dense2')  # [-1, 2]
        return state


class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.price = np.random.normal(size=[config.batch_size, config.days])
        self.index = 0

    def num(self):
        return self.config.days - self.config.num_step

    def next_batch(self):
        x = self.price[:, self.index: self.index + self.config.num_step]  # [batch_size, num_step]
        x = np.transpose(x, [1, 0])  # [num_step, batch_size]
        y = self.price[:, self.index + self.config.num_step]  # [batch_size]
        self.index = (self.index + 1) % self.num()
        return x, y


class Stock:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.tensors = Tensors(config)
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

        self.samples = Samples(cfg)
        for epoch in range(cfg.epoches):
            for batch in range(self.samples.num() // cfg.gpus):
                feed_dict = {
                    self.tensors.lr: cfg.lr
                }
                for gpu_index in range(cfg.gpus):
                    x, y = self.samples.next_batch()
                    feed_dict[self.tensors.sub_ts[gpu_index].x] = x
                    feed_dict[self.tensors.sub_ts[gpu_index].y] = y
                _, loss, su = self.session.run([self.tensors.train_op, self.tensors.loss, self.tensors.summary_op],
                                               feed_dict)
                writer.add_summary(su, epoch * self.samples.num() + batch)
                print('%d/%d: loss=%.8f' % (batch, epoch, loss))
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path)

    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()
    stock = Stock(config)
    stock.train()
    stock.close()
    print('Finished!')
