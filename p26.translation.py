import tensorflow as tf
import numpy as np
import os
import argparse


class Config:
    def __init__(self):
        self.batch_size = 20
        self.num_step1 = 8
        self.num_step2 = 10
        self.num_units = 5

        self.gpus = self.get_gpus()
        self.ch_size = 200
        self.en_size = 100

        self.name = 'p26'
        self.save_path = 'models/{name}/{name}'.format(name=self.name)
        self.logdir = 'logs/{name}/'.format(name=self.name)

        self.lr = 0.0002
        self.epoches = 200

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
    def __init__(self, config: Config):
        self.config = config
        self.sub_tensors = []
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('translation'):
            for gpu_index in range(config.gpus):
                self.sub_tensors.append(Sub_tensors(config, gpu_index, opt))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            self.grad = self.merge_grads()
            self.train_op = opt.apply_gradients(self.grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_tensors])
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            self.show_paras()

    def show_paras(self):
        total = 0
        for var in tf.trainable_variables():
            num = self.get_num(var.shape)
            print(var.name, var.shape, num)
            total += num
        print('total var:', total)

    def get_num(self, shape):
        num = 1
        for v in shape:
            num *= v
        return num








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

        results = [(tf.reduce_mean(grads[v], axis = 0),v) for v in grads]
        for v in indexed_grads:
            indices = tf.concat([g.indices for g in indexed_grads[v]], axis = 0)
            values = tf.concat([g.values for g in indexed_grads[v]], axis=0)
            g = tf.IndexedSlices(values, indices)
            results.append((g, v))
        return results

class Sub_tensors:
    def __init__(self, config: Config, gpu_index, opt: tf.train.AdamOptimizer):
        self.config = config
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.int32, [None, config.num_step1], 'x')
            self.y = tf.placeholder(tf.int32, [None, config.num_step2], 'y')
            with tf.variable_scope('encode'):
                state = self.encode()
            with tf.variable_scope('decode'):
                loss_list, self.y_predict = self.decode(state)
            self.loss = tf.reduce_mean(loss_list)
            self.grad = opt.compute_gradients(self.loss)

    def encode(self):
        config = self.config
        batch_size = tf.shape(self.x)[0]

        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name = 'encode_lstm1')
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='encode_lstm2')
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        char_dict = tf.get_variable('char_dict', [config.ch_size, config.num_units], tf.float32)
        x = tf.nn.embedding_lookup(char_dict, self.x)#-1, num_stpe1, num_units

        state = cell.zero_state(batch_size, tf.float32)
        for i in range(config.num_step1):
            xi = x[:, i]
            _, state = cell(xi, state)
        return state #((c,h), (c,h))

    def decode(self, state):
        config = self.config
        batch_size = tf.shape(self.x)[0]

        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name = 'decode_lstm1')
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='decode_lstm2')
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        y = tf.one_hot(self.y, config.en_size)#-1, num_step2, en_size
        xi = tf.zeros([batch_size, config.num_units], tf.float32)

        loss_list = []
        y_list = []
        with tf.variable_scope('rnn'):
            for i in range(config.num_step2):
                y_pred, state = cell(xi, state)
                y_pred = tf.layers.dense(y_pred, config.en_size, name = 'y_predict')
                tf.get_variable_scope().reuse_variables()

                yi = y[:, i]#_1, en_size
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=yi)
                loss_list.append(loss)

                y_list.append(tf.math.argmax(y_pred, axis = 1, output_type=tf.int32))
        return loss_list, y_list






class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0
        self.chinese = list(np.random.randint(0, config.ch_size, [self.num(), config.num_step1]))
        self.english = list(np.random.randint(0, config.en_size, [self.num(), config.num_step2]))

    def num(self):
        return 100

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.num():
            x = self.chinese[self.index: next]
            y = self.english[self.index: next]
        else:
            x = self.chinese[self.index:]
            y = self.english[self.index:]
            next -= self.num()
            x += self.chinese[:next]
            y += self.english[:next]
        self.index = next
        return x, y

class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.samples = Samples(config)
            self.session = tf.Session(config = conf, graph=graph)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph=graph)
            self.tensors = Tensors(config)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, config.save_path)
            except:
                self.session.run(tf.global_variables_initializer())

    def train(self):
        cfg = self.config
        batches = self.samples.num() // (cfg.gpus * cfg.batch_size)
        for epoch in range(cfg.epoches):
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: cfg.lr
                }
                for gpu_index in range(cfg.gpus):
                    x, y = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = x
                    feed_dict[self.tensors.sub_tensors[gpu_index].y] = y
                _, loss, su = self.session.run([self.tensors.train_op,
                                      self.tensors.loss,
                                      self.tensors.summary_op], feed_dict)
                self.file_writer.add_summary(su, epoch * batches + batch)
                print('%d/%d, loss = %f' % (epoch, batch, loss))
            self.saver.save(self.session, config.save_path)

    def close(self):
        self.session.close()


    def predict(self, sentence):
        feed_dict = {
            self.tensors.sub_tensors[0].x: sentence
        }
        predict = self.session.run(self.tensors.sub_tensors[0].y_predict, feed_dict)
        return predict



if __name__ == '__main__':
    config = Config()
    app = App(config)
    # app.train()

    sentence = np.random.randint(0, config.ch_size, [2, config.num_step1])
    r = app.predict(sentence)
    r = np.array(r)
    print(r[:, 0])


    app.close()

