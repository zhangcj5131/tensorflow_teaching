import tensorflow as tf
import numpy as np
import argparse
import os

class Config:
    def __init__(self):
        #多少只股票
        self.stock_num = 10
        #用多少天的数据预测
        self.num_step = 8
        self.state_size = 2
        self.hidden_size = 7
        self.days = 100
        self.name = 'p22'
        self.save_path = 'models/{name}/{name}'.format(name=self.name)
        self.logdir = 'logs/{name}/'.format(name=self.name)
        self.lr = 0.0002
        self.epoches = 10
        self.gpu = self._get_gpu()


    def _get_gpu(self):
        value = os.getenv('CUDA_VISIBLE_DEVICES', '0,4')
        return len(value.split(','))

    def from_cmd_line(self):
        attrs = self._get_attrs()
        parse = argparse.ArgumentParser()
        for name, value in attrs.items():
            print('add %s from cmd' % name)
            parse.add_argument('--'+name, default=value, type=type(value), help='default=%s' % value)
        a = parse.parse_args()
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
        self.sub_tensors = []
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('stock'):
            for gpu_index in range(config.gpu):
                self.sub_tensors.append(Sub_tensor(gpu_index, config, opt))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grads = self.merge_grad()
            self.train_op = opt.apply_gradients(grads)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_tensors])
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def merge_grad(self):
        grads = {}
        for ts in self.sub_tensors:
            for g, v in ts.grad:
                if v not in grads:
                    grads[v] = []
                grads[v].append(g)

        # for v in grads:
        #     print(grads[v])
        #[(g, v), (g, v)-------]
        result_grads = [(tf.reduce_mean(grads[v], axis = 0), v) for v in grads]
        return result_grads





class Sub_tensor:
    def __init__(self, gpu_index, config: Config, opt: tf.train.AdamOptimizer):
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.float32, [config.num_step, None], 'x')  # num_step, stock_num
            self.y = tf.placeholder(tf.float32, [None], 'y')  # stock_num

            cell = MyCell(config.hidden_size, config.state_size)
            # state = cell.zero_state(config.stock_num)
            state = cell.zero_state(tf.shape(self.x)[1])

            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    xi = self.x[i]  # [stock_num]
                    xi = tf.reshape(xi, [-1, 1])  # [stock_num, 1]
                    state = cell(xi, state, 'my_cell')
                    tf.get_variable_scope().reuse_variables()

            y_predict = tf.layers.dense(state, 1, name='dense')  # stock_num, 1
            self.y_predict = tf.reshape(y_predict, [-1])  # stock_num

            self.loss = tf.reduce_mean(tf.abs(self.y - self.y_predict))
            #[(g, v), (g, v)-------]
            self.grad = opt.compute_gradients(self.loss)
            # print(len(self.grad))



class MyCell:
    def __init__(self, hidden_size, state_size):
        self.hidden_size = hidden_size
        self.state_size = state_size

    def zero_state(self, stock_num, dtype=tf.float32):
        return tf.zeros(shape = [stock_num, self.state_size], dtype=dtype)

    #xi:[stock_num, 1]
    #state:[stock_num, state_size]
    def __call__(self, xi, state, name):
        with tf.variable_scope(name):
            xi = tf.concat([xi, state], axis=1)#stock_num, state_zie + 1
            xi = tf.layers.dense(xi, self.hidden_size, tf.nn.relu, name = 'dense1')#stock_num, hidden_size
            state = tf.layers.dense(xi, self.state_size, name = 'dense2')#stock_num, state_size
            return state

class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.price = np.random.normal(size = [config.stock_num, config.days])
        self.index = 0

    def num(self):
        return self.config.days - self.config.num_step

    def next_batch(self):
        x = self.price[:, self.index: self.index + self.config.num_step]#stock_num, num_step
        x = np.transpose(x, [1, 0])
        y = self.price[:, self.index + self.config.num_step]#[stock_num]
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

        cfg.stock_num = 14
        self.samples = Samples(config)
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)


        for epoch in range(cfg.epoches):
            for batch in range(self.samples.num()):

                feed_dict = {
                    self.tensors.lr: cfg.lr,
                }
                for gpu_index in range(config.gpu):
                    x, y = self.samples.next_batch()
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = x
                    feed_dict[self.tensors.sub_tensors[gpu_index].y] = y

                _, loss, su= self.session.run([self.tensors.train_op, self.tensors.loss, self.tensors.summary_op],
                                               feed_dict)
                writer.add_summary(su, epoch * self.samples.num() + batch)
                #print('%d/%d: loss=%.8f' % (batch, epoch, loss))
                print('%d/%d: %d' % (batch, epoch, y.shape[0]))
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path)


    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()
    stock = Stock(config)
    try:
        stock.train()
    finally:
        stock.close()
    print('Finished!')