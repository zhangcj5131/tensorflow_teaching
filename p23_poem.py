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
                self.sub_tensors.append(SubTensors(config, gpu_index, opt, char_size))
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
        index_grads = {}
        grads = {}
        for ts in self.sub_tensors:
            for g, v in ts.grad:
                if isinstance(g, tf.IndexedSlices):
                    if v not in index_grads:
                        index_grads[v] = []
                    index_grads[v].append(g)
                else:
                    if v not in grads:
                        grads[v] = []
                    grads[v].append(g)
        result = [(tf.reduce_mean(grads[v], axis = 0), v) for v in grads]

        for v in index_grads:
            indices = tf.concat([g.indices for g in index_grads[v]], axis = 0)
            values = tf.concat([g.values for g in index_grads[v]], axis = 0)
            g = tf.IndexedSlices(values, indices)
            result.append((g, v))
        return result

class SubTensors:
    def __init__(self, config: Config, gpu_index, opt: tf.train.AdamOptimizer, char_size):
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.int32, [None, config.num_step], 'x')#-1, 32
            char_dict = tf.get_variable('char_dict', [char_size, config.num_units], tf.float32)#4000, 200
            x = tf.nn.embedding_lookup(char_dict, self.x) #-1, 32, 200

            cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name = 'lstm1')
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='lstm2')

            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

            batch_size_tf = tf.shape(self.x)[0]
            state = cell.zero_state(batch_size_tf, tf.float32)

            y = tf.concat([self.x[:, 1:], tf.zeros([batch_size_tf, 1], tf.int32)], axis = 1)
            y_one_hot = tf.one_hot(y, char_size)

            loss_list = []
            precise_list = []
            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    xi = x[:, i]
                    y_pre, state = cell(xi, state)

                    y_pre = tf.layers.dense(y_pre, char_size, name = 'y_dense')

                    yi = y_one_hot[:, i]

                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(yi, y_pre)
                    loss_list.append(loss)

                    y_predict = tf.math.argmax(y_pre, axis = 1, output_type=tf.int32)
                    y_label = y[:, i]
                    precise = tf.equal(y_predict, y_label)
                    precise = tf.cast(precise, tf.float32)
                    precise = tf.reduce_mean(precise)
                    precise_list.append(precise)
                    tf.get_variable_scope().reuse_variables()


        self.loss = tf.reduce_mean(loss_list)
        self.precise = tf.reduce_mean(precise_list)
        self.grad = opt.compute_gradients(self.loss)

        self.batch_size = tf.placeholder(tf.int32, [], 'batch_size')
        self.xi = tf.placeholder(tf.int32, [None], 'xi')
        self.input_state = cell.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope('rnn', reuse=True):
            xi = tf.nn.embedding_lookup(char_dict, self.xi)
            y_pred, self.state = cell(xi, self.input_state)
            y_pred = tf.layers.dense(y_pred, char_size, name = 'y_dense')
            self.y_predict = tf.math.argmax(y_pred, axis = 1)



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

    def get_random_char(self):
        id = np.random.randint(0, self.char_size)
        return self.id_char_dict[id]

    def encode(self, head):
        return [self.char_id_dict[ch] for ch in head]

    def decode(self, char_list):
        # print(char_list)
        result = ''
        # print(self.id_char_dict)
        # print(self.char_id_dict)
        # print(self.id_char_dict[1630])
        # print(self.id_char_dict[5846])
        for id in char_list:
            # print(id, self.id_char_dict[id])
            result += self.id_char_dict[id]
        return result


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
            self.tensors = Tensors(config, self.samples.char_size)
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

    def predict(self, head):
        if head is None:
            head = self.samples.get_random_char()
        head = self.samples.encode(head)

        result = head
        state = self.session.run([self.tensors.sub_tensors[0].input_state], {self.tensors.sub_tensors[0].batch_size:1})
        y_predict = None
        for i in range(config.num_step):
            xi = [head[i]] if i < len(head) else y_predict
            if i >= len(head):
                result.append(y_predict[0])
            feed_dict = {
                self.tensors.sub_tensors[0].xi: xi,
                self.tensors.sub_tensors[0].input_state: state
            }
            y_predict, state = self.session.run([self.tensors.sub_tensors[0].y_predict,
                                                 self.tensors.sub_tensors[0].state], feed_dict)
        # print(result)
        result = self.samples.decode(result)
        return result







if __name__ == '__main__':
    config = Config()
    # s = Samples(config)
    # t = Tensors(config, s.size)
    poem = Poem(config)
    result = poem.predict('我')
    print(result)
