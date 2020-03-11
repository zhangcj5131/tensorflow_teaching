import tensorflow as tf
import numpy as np
import argparse
import os


class Config:
    def __init__(self):
        self.batch_size = 2
        self.num_step = 8 * 4
        self.num_units = 200
        self.gpus = self.get_gpus()

        self.sample_path = 'qts_7X4.txt'

        self.name = 'p23'
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
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('poem'):
            for gpu_index in range(config.gpus):
                self.sub_tensors.append((Sub_tensors(config, gpu_index, opt, char_size)))
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
        indexed_grads = {}
        grads = {}
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
            values = tf.concat([g.values for g in indexed_grads[v]], axis = 0)
            g = tf.IndexedSlices(values, indices)
            result.append((g, v))
        return  result



class Sub_tensors:
    def __init__(self, config: Config, gpu_index, opt: tf.train.AdamOptimizer, char_size):
        self.x = tf.placeholder(tf.int32, [None, config.num_step], 'x')#-1, 32
        char_dict = tf.get_variable('char_dict', [char_size, config.num_units], tf.float32)
        x = tf.nn.embedding_lookup(char_dict, self.x)

        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name = 'lstm1')
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name='lstm2')

        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        batch_size_tf = tf.shape(self.x)[0]
        state = cell.zero_state(batch_size_tf, tf.float32)

        y = tf.concat([self.x[:, 1:], tf.zeros([batch_size_tf, 1], tf.int32)], axis = 1)#-1, 32
        y_onehot = tf.one_hot(y, char_size)

        loss_list = []
        precise_list = []

        for i in range(config.num_step):
            xi = x[:, i]
            yi_pre, state = cell(xi, state)
            yi_pre = tf.layers.dense(yi_pre, char_size, name = 'y_dense')
            yi = y_onehot[:, i]

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yi, logits=yi_pre)
            loss_list.append(loss)

            yi_predict = tf.math.argmax(yi_pre, axis = 1, output_type=tf.int32)
            yi_label = y[:, i]
            precise = tf.reduce_mean(tf.cast(tf.equal(yi_label, yi_predict), tf.float32))
            precise_list.append(precise)

            tf.get_variable_scope().reuse_variables()

        self.loss = tf.reduce_mean(loss_list)
        self.precise = tf.reduce_mean(precise_list)
        self.grad = opt.compute_gradients(self.loss)


class Chinese:
    def __init__(self, config: Config):
        self.config = config
        f = open(config.sample_path, encoding='utf-8')
        with f:
            poems = f.readlines()
        result = []
        char_set = set()
        for poem in poems:
            poem = poem.rstrip()
            result.append(poem)
            for ch in poem:
                char_set.add(ch)

        self.char_id_dict = {}
        self.id_char_dict = {}
        for id, ch in enumerate(char_set):
            self.char_id_dict[ch] = id
            self.id_char_dict[id] = ch
        self.poems = [[self.char_id_dict[ch] for ch in poem] for poem in result]

    @property
    def char_size(self):
        return len(self.char_id_dict)


class Samples(Chinese):
    def __init__(self, config: Config):
        super(Samples, self).__init__(config)
        self.config = config
        self.index = 0

    @property
    def size(self):
        return len(self.poems)

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.num:
            x = self.poems[self.index: next]
        else:
            x = self.poems[self.index:]
            next -= self.num
            x += self.poems[:next]
        self.index = next
        return x


class Poem:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.samples = Samples(config)
            self.tensors = Tensors(config, self.samples.size)
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            self.file_writer = tf.summary.FileWriter(config.logdir, graph=graph)
            try:
                self.saver.restore(self.session, config.save_path)
                print('Restore the model from %s successfully.' % config.save_path)
            except:
                print('Fail to restore the model from', config.save_path)
                self.session.run(tf.global_variables_initializer())

    def train(self):
        cfg = self.config
        batches = self.samples.num // (cfg.gpus * cfg.batch_size)
        for epoch in range(cfg.epoches):
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: cfg.lr
                }
                for gpu_index in range(cfg.gpus):
                    x = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = x
                    _, precise, lo, su = self.session.run([self.tensors.train_op,
                                      self.tensors.precise,
                                      self.tensors.loss,
                                      self.tensors.summary_op], feed_dict)
                    self.file_writer.add_summary(su, epoch*batches + batch)
                    print('%d/%d, loss = %f, precise = %f' % (epoch, batch, lo, precise))
            self.saver.save(self.session, cfg.save_path)




if __name__ == '__main__':
    config = Config()
    s = Samples(config)
    t = Tensors(config, s.size)
    # c = Samples(config)
    # print(c.next_batch(2))
    # c = Chinese(config)
    # sample = Samples(config)
    # char_size = sample.size
    #
    # tensor = Tensors(config, char_size)
    # r = sample.next_batch(10)
    # print(r)
    # r = sample.next_batch(10)
    # print(r)

    # print(tf.__version__)
