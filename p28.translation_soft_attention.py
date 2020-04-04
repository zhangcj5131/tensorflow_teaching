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
        self.level = 2

        self.gpus = self.get_gpus()
        self.ch_size = 200
        self.en_size = 100

        self.name = 'p28'
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

            losses, self.y_predict = self.encode_decode()
            self.loss = tf.reduce_mean(losses)
            self.grad = opt.compute_gradients(self.loss)

    def encode_decode(self):
        with tf.variable_scope('encode'):
            state, zlist = self.encode(self.x)
        with tf.variable_scope('decode'):
            losses, y_predict = self.decode(state, zlist, self.y)
        return losses, y_predict

    def encode(self, x):
        config = self.config
        char_dict = tf.get_variable('char_dict', [config.ch_size, config.num_units], tf.float32)
        x = tf.nn.embedding_lookup(char_dict, x)#-1, num_step1, num_units

        cell = self.get_cell()
        batch_size = tf.shape(x)[0]
        state = cell.zero_state(batch_size, tf.float32)

        zlist = []
        for i in range(config.num_step1):
            xi = x[:, i]
            zi, state = cell(xi, state)
            zlist.append(zi)
        #zlist:step1, -1, units
        zlist = tf.transpose(zlist, [1, 0, 2])#-1, step1, units
        return state, zlist

    def get_cell(self):
        cells = []
        for i in range(self.config.level):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_units, name = 'lstm_%d' % i)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell


    def decode(self, state, zlist, y):
        config = self.config
        cell = self.get_cell()
        y = tf.one_hot(y, config.en_size)#-1, step2, en_size

        losses = []
        y_list = []
        for i in range(config.num_step2):
            xi = self.get_soft_attention(state, zlist)
            y_pred, state = cell(xi, state)
            y_pred = tf.layers.dense(y_pred, config.en_size, name = 'y_predict')
            tf.get_variable_scope().reuse_variables()

            yi = y[:, i]
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=yi)
            losses.append(loss)
            y_list.append(tf.math.argmax(y_pred, axis = 1, output_type=tf.int32))
        return losses, y_list

    #state:(LSTMStateTuple(c, h), LSTMStateTuple(c, h))
    #z_list:-1, step1, units
    def get_soft_attention(self, state, zlist):
        config = self.config
        state = self.convert_state(state)#[-1, 4*num_units]
        alpha = tf.reshape(zlist, [-1, config.num_step1*config.num_units])
        alpha = tf.concat([state, alpha], axis = 1)#[-1, num_step1*num_units+4*num_units]
        with tf.variable_scope('attention'):
            alpha = tf.layers.dense(alpha, config.num_step1, activation=tf.nn.relu, name = 'dense1')#-1, num_step1
            alpha = tf.layers.dense(alpha, config.num_step1, name='dense2')#-1, num_step1
            alpha = tf.nn.softmax(alpha, axis = 1)#-1, num_step1
            alpha = tf.reshape(alpha, [-1, config.num_step1, 1])
            attention = alpha*zlist # -1, step1, units
            attention = tf.reduce_sum(attention, axis = 1)#-1, num_units
            return attention

    def convert_state(self, state):
        result = []
        for lstm_state_tuple in state:
            result.append(lstm_state_tuple.c)
            result.append(lstm_state_tuple.h)
        #result:4, -1, units
        result = tf.transpose(result, [1, 0, 2])#-1, 4, units
        result = tf.reshape(result, [-1, 4*self.config.num_units])
        return result







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
    app.train()

    # sentence = np.random.randint(0, config.ch_size, [2, config.num_step1])
    # r = app.predict(sentence)
    # r = np.array(r)
    # print(r[:, 0])


    app.close()

