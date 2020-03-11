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
    def __init__(self, config: Config, ch_size):
        self.config = config
        self.sub_ts = []

        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('poem'):
            for gpu_index in range(config.gpus):
                self.sub_ts.append(SubTensors(config, gpu_index, opt, ch_size))
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
        # print(len(indexed_grads))
        # print(len(grads))
        grads = [(tf.reduce_mean(grads[v], axis=0), v) for v in grads]

        for v in indexed_grads:
            indices = tf.concat([g.indices for g in indexed_grads[v]], axis=0)
            values = tf.concat([g.values for g in indexed_grads[v]], axis=0)
            g = tf.IndexedSlices(values, indices)
            grads.append((g, v))

        return grads


class SubTensors:
    def __init__(self, config, gpu_index, opt, ch_size):
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.int32, [None, config.num_step], 'x')

            # word2vec
            ch_dict = tf.get_variable('ch_dict', [ch_size, config.num_units], tf.float32)
            x = tf.nn.embedding_lookup(ch_dict, self.x)  # [-1, num_step, num_units]

            cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name='lstm1')
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name='lstm2')
            # peephole_cell = tf.nn.rnn_cell.LSTMCell(use_peepholes=True)

            # cell = MyLSTMCell(config.num_units, 'lstm')
            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
            # cell = MyMultiRNNCell([cell] * 2)

            batch_size_ts = tf.shape(self.x)[0]
            state = cell.zero_state(batch_size_ts, tf.float32)  # [batch_size, state_size]
            # print(state)
            # state = [tf.nn.rnn_cell.LSTMStateTuple, tf.nn.rnn_cell.LSTMStateTuple]

            y0 = tf.concat((self.x[:, 1:], tf.zeros([batch_size_ts, 1], tf.int32)), axis=1)  # [-1, num_step]
            y = tf.one_hot(y0, ch_size)  # [-1, num_step, ch_size]
            losses = []
            precises = []
            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    xi = x[:, i, :]  # [-1, num_units]
                    yi_pred, state = cell(xi, state)  # [-1, num_units]

                    yi_pred = tf.layers.dense(yi_pred, ch_size, name='yi_dense')  # [-1, ch_size]
                    yi = y[:, i, :]  # [-1, ch_size]

                    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yi, logits=yi_pred)
                    losses.append(loss_i)
                    tf.get_variable_scope().reuse_variables()

                    yi_pred0 = tf.argmax(yi_pred, axis=1, output_type=tf.int32)  # [-1]
                    yi0 = y0[:, i]  # [-1]
                    precise = tf.equal(yi_pred0, yi0)
                    precise = tf.cast(precise, tf.float32)
                    precise = tf.reduce_mean(precise)
                    precises.append(precise)

            self.loss = tf.reduce_mean(losses)
            self.grad = opt.compute_gradients(self.loss)  # [(g1, v1), (g2, v2), ... , (gn, vn)]
            self.precise = tf.reduce_mean(precises)

            self.batch_size = tf.placeholder(tf.int32, [], 'batch_size')
            self.zero_state = cell.zero_state(self.batch_size, tf.float32)
            self.xi = tf.placeholder(tf.int32, [None], 'xi')
            self.input_state = self.zero_state
            with tf.variable_scope('rnn', reuse=True):
                xi = tf.nn.embedding_lookup(ch_dict, self.xi)  # [-1, num_units]
                yi_predict, self.state = cell(xi, self.input_state)
                yi_predict = tf.layers.dense(yi_predict, ch_size, name='yi_dense')
                self.yi_predict = tf.argmax(yi_predict, axis=1)


class MyMultiRNNCell:
    def __init__(self, cells):
        self.cells = cells

    def zero_state(self, batch_size, dtype=tf.float32):
        return [cell.zero_state(batch_size, dtype) for cell in self.cells]

    def __call__(self, xi, state):
        new_states = []
        name_id = 0
        for st, cell in zip(state, self.cells):
            with tf.variable_scope('cell_%d' % name_id):
                xi, new_state = cell(xi, st)
            name_id += 1
            new_states.append(new_state)
        return xi, new_states


class MyLSTMCell:
    def __init__(self, num_units, name):
        self.num_units = num_units
        self.name = name

    def zero_state(self, batch_size, dtype=tf.float32):
        c = tf.zeros([batch_size, self.num_units], dtype)
        h = tf.zeros([batch_size, self.num_units], dtype)
        # tf.nn.rnn_cell.LSTMStateTuple
        return [c, h]

    def __call__(self, xi, state):
        """

        :param xi: [-1, num_units]
        :param state: [[-1, num_units], [-1, num_units]]
        :return: [-1, num_units], [c, h]
        """

        with tf.variable_scope(self.name):
            forget_gate = self._get_gate(xi, state[1], 'forget')
            input_gate = self._get_gate(xi, state[1], 'input')
            output_gate = self._get_gate(xi, state[1], 'output')

            xi = tf.nn.tanh(self._full_connect(xi, state[1], name='input_fc'))
            c = forget_gate * state[0] + input_gate * xi
            h = output_gate * tf.nn.tanh(c)
        return h, [c, h]

    def _get_gate(self, xi, hi, name):
        with tf.variable_scope(name):
            return tf.nn.sigmoid(self._full_connect(xi, hi, 'fc'))

    def _full_connect(self, xi, hi, name):
        with tf.variable_scope(name):
            xi = tf.concat((xi, hi), axis=1)  # [-1, 2 * num_units]
            return tf.layers.dense(xi, self.num_units, name='dense')


class Chinese:
    def __init__(self, config: Config):
        f = open(config.sample_path, encoding='utf-8')
        with f:
            result = []
            s = set()
            poems = f.readlines()

        for poem in poems:
            poem = poem.rstrip()
            # print(len(poem))
            result.append(poem)
            for ch in poem:
                s.add(ch)
        self.ch_map = {}
        for id, ch in enumerate(sorted(s)):
            self.ch_map[ch] = id

        self.poems = [[self.ch_map[ch] for ch in poem] for poem in result]

    @property
    def size(self):
        return len(self.ch_map)

    def get_random_ch(self):
        id = np.random.randint(0, self.size)
        return self.get_ch(id)

    def get_ch(self, id):
        for ch in self.ch_map:
            if self.ch_map[ch] == id:
                return ch

    def encode(self, ch_str):
        result = [self.ch_map[ch] for ch in ch_str]
        return result

    def decode(self, code_list):
        result = ''
        for code in code_list:
            result += self.get_ch(code)
        return result


class Samples(Chinese):
    def __init__(self, config: Config):
        super(Samples, self).__init__(config)
        self.config = config
        self.index = 0

    def num(self):
        """
        The number of the samples in total.
        :return:
        """
        return len(self.poems)

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.num():
            result = self.poems[self.index: next]
        else:
            result = self.poems[self.index:]
            next -= self.num()
            result += self.poems[:next]
        self.index = next
        return result  # [batch_size, num_step]


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
                    self.tensors.lr: cfg.lr
                }
                for gpu_index in range(cfg.gpus):
                    x = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_ts[gpu_index].x] = x
                _, loss, su, precise = self.session.run(
                    [self.tensors.train_op, self.tensors.loss, self.tensors.summary_op, self.tensors.precise], feed_dict)
                writer.add_summary(su, epoch * batches + batch)
                print('%d/%d: loss=%.8f, precise=%.8f' % (batch, epoch, loss, precise), flush=True)
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path, flush=True)

    def predict(self, head=None):
        if head is None:
            head = self.samples.get_random_ch()  # get a random chinese character
        head = self.samples.encode(head)  # Encode the characters into ids
        result = head
        state = self.session.run(self.tensors.sub_ts[0].zero_state, {self.tensors.sub_ts[0].batch_size: 1})

        for i in range(self.config.num_step):
            xi = [head[i]] if i < len(head) else yi_predict
            feed_dict = {
                self.tensors.sub_ts[0].xi: xi,
                self.tensors.sub_ts[0].input_state: state
            }
            if i >= len(head):
                result.append(yi_predict[0])
            yi_predict, state = self.session.run([self.tensors.sub_ts[0].yi_predict, self.tensors.sub_ts[0].state], feed_dict)
        result = self.samples.decode(result)
        return result

    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()

    # ch = Chinese(config)
    # for poem in ch.poems:
    #     print(poem)

    poem = Poem(config)

    # print(poem.predict('我是中国人'))

    poem.train()
    poem.close()
    print('Finished!')
