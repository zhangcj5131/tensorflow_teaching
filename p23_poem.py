import tensorflow as tf
import numpy as np
import argparse
import os


class Config:
    def __init__(self):
        self.batch_size = 50
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
                self.sub_tensors.append(Sub_tensors(config, gpu_index, opt, char_size))
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
            values = tf.concat([g.values for g in indexed_grads[v]], axis = 0)
            g = tf.IndexedSlices(values, indices)
            result.append((g, v))

        return result

class Sub_tensors:
    def __init__(self, config: Config, gpu_index, opt: tf.train.AdamOptimizer, char_size):
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.int32, [None, config.num_step], 'x') #-1, 32
            char_dict = tf.get_variable('char_dict', [char_size, config.num_units], tf.float32)#4000, 200
            x = tf.nn.embedding_lookup(char_dict, self.x)#-1, 32, 200



            cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name = 'lstm1')
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, state_is_tuple=True, name='lstm2')
            cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

            # cell1 = MyLSTM(config.num_units, name = 'lstm1')
            # cell2 = MyLSTM(config.num_units, name='lstm2')
            # cell = MyMultiLSTM([cell1, cell2])






            batch_size_tf = tf.shape(self.x)[0]
            state = cell.zero_state(batch_size_tf, tf.float32)

            y = tf.concat([self.x[:, 1:], tf.zeros([batch_size_tf, 1], tf.int32)], axis = 1)#-1, 32
            y_onehot = tf.one_hot(y, char_size)#-1, 32, 4000

            loss_list = []
            precise_list = []
            with tf.variable_scope('rnn'):
                for i in range(config.num_step):
                    xi = x[:, i]#-1, 200
                    yi_pred, state = cell(xi, state)
                    yi_pred = tf.layers.dense(yi_pred, char_size, name = 'yi_predict')#-1, char_size
                    yi_onehot = y_onehot[:, i]#-1, 4000

                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=yi_pred, labels=yi_onehot)
                    loss_list.append(loss)

                    yi_predict_id = tf.math.argmax(yi_pred, axis = 1, output_type=tf.int32)
                    yi_label_id = y[:, i]
                    precise = tf.equal(yi_label_id, yi_predict_id)
                    precise = tf.cast(precise, tf.float32)
                    precise = tf.reduce_mean(precise)
                    precise_list.append(precise)
                    tf.get_variable_scope().reuse_variables()
            self.loss = tf.reduce_mean(loss_list)
            self.precise = tf.reduce_mean(precise_list)
            self.grad = opt.compute_gradients(self.loss)


            self.batch_size = tf.placeholder(tf.int32, [], 'batch_size')
            self.xi = tf.placeholder(tf.int32, [None], 'xi')
            self.inputstate = cell.zero_state(self.batch_size, tf.float32)
            with tf.variable_scope('rnn', reuse=True):
                xi = tf.nn.embedding_lookup(char_dict, self.xi)
                y_predict, self.state = cell(xi, self.inputstate)
                y_predict = tf.layers.dense(y_predict, char_size, name = 'yi_predict')
                self.y_predict = tf.math.argmax(y_predict, axis = 1)




class MyMultiLSTM:
    def __init__(self, cells):
        self.cells = cells

    def zero_state(self, batch_size, dtype = tf.float32):
        state = [cell.zero_state(batch_size, dtype) for cell in self.cells]#[(c,h), (c,h), (c, h)----]
        return state

    def __call__(self, xi, state):
        new_states = []
        id = 0
        for st, cell in zip(state, self.cells):
            with tf.variable_scope('cell_%d' % id):
                xi, new_state = cell(xi, st)
            id += 1
            new_states.append(new_state)
        return xi, new_states




class MyLSTM:
    def __init__(self, num_units, name):
        self.num_units = num_units
        self.name = name

    def zero_state(self, batch_size, dtype=tf.float32):
        c =  tf.zeros([batch_size, self.num_units], dtype)
        h = tf.zeros([batch_size, self.num_units], dtype)
        return [c, h]

    def __call__(self, xi, state):
        with tf.variable_scope(self.name):
            intput_gate = self._gate(xi, state[1], 'intput_gate')
            output_gate = self._gate(xi, state[1], 'output_gate')
            forget_gate = self._gate(xi, state[1], 'forget_gate')

            xi = tf.nn.tanh(self._full_connect(xi, state[1], 'fc_input'))
            c = forget_gate * state[0] + intput_gate * xi
            h = output_gate * tf.nn.tanh(c)

            new_state = [c, h]
            return h, new_state

    def _gate(self, c, h, name):
        with tf.variable_scope(name):
            return tf.nn.sigmoid(self._full_connect(c, h, 'fc'))

    def _full_connect(self, c, h, name):
        with tf.variable_scope(name):
            x = tf.concat([c, h], axis = 1)
            return tf.layers.dense(x, self.num_units, name = 'dense')










class Chinese:
    def __init__(self, config: Config):
        self.config = config
        f = open(config.sample_path, encoding='utf-8')
        with f:
            poems = f.readlines()
        char_set = set()
        result = []
        for poem in poems:
            poem = poem.rstrip()
            result.append(poem)
            for ch in poem:
                char_set.add(ch)
        self.char_id_dict = {}
        self.id_char_dict = {}
        for id, char in enumerate(sorted(char_set)):
            self.char_id_dict[char] = id
            self.id_char_dict[id] = char

        self.poems = [[self.char_id_dict[ch] for ch in poem] for poem in result]


    def get_random_char(self):
        id = np.random.randint(0, self.char_size)
        return self.id_char_dict[id]

    def encode(self, char_list):
        return [self.char_id_dict[ch] for ch in char_list]

    def decode(self, id_list):
        result = ''
        for id in id_list:
            result += self.id_char_dict[id]
        return result


    @property
    def char_size(self):
        return len(self.char_id_dict)

class Samples(Chinese):
    def __init__(self, config: Config):
        super(Samples, self).__init__(config)
        self.index = 0

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.poem_size:
            result = self.poems[self.index: next]
        else:
            result = self.poems[self.index:]
            next -= self.poem_size
            result += self.poems[:next]
        self.index = next
        return result

    @property
    def poem_size(self):
        return len(self.poems)



class Poem:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config=conf, graph=graph)
            self.samples = Samples(config)
            self.tensors = Tensors(config, self.samples.char_size)
            self.file_write = tf.summary.FileWriter(logdir=config.logdir, graph=graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, config.save_path)
            except:
                self.session.run(tf.global_variables_initializer())

    def train(self):
        config = self.config
        batches = self.samples.poem_size // (config.gpus * config.batch_size)
        for epoch in range(config.epoches):
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr:config.lr
                }
                for gpu_index in range(config.gpus):
                    x = self.samples.next_batch(config.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = x
                _, loss, precise, su = self.session.run([self.tensors.train_op,
                              self.tensors.loss,
                              self.tensors.precise,
                              self.tensors.summary_op], feed_dict)
                self.file_write.add_summary(su, epoch*batches + batch)
                print('%d/%d, loss = %f, precise = %f' % (epoch, batch, loss, precise))
            self.saver.save(self.session, config.save_path)

    def close(self):
        self.session.close()

    def predict(self, head=None):
        if head is None:
            head = self.samples.get_random_char()
        head = self.samples.encode(head)
        result = head

        state = self.session.run(self.tensors.sub_tensors[0].inputstate, {self.tensors.sub_tensors[0].batch_size:1})
        y_predict = None
        for i in range(self.config.num_step):
            xi = [head[i]] if i < len(head) else y_predict

            feed_dict = {
                self.tensors.sub_tensors[0].xi: xi,
                self.tensors.sub_tensors[0].inputstate: state
            }
            if i >= len(head):
                result.append(y_predict[0])
            y_predict, state = self.session.run([self.tensors.sub_tensors[0].y_predict,
                                                 self.tensors.sub_tensors[0].state], feed_dict)
        result = self.samples.decode(result)
        return result







if __name__ == '__main__':
    config = Config()
    s = Samples(config)
    # Tensors(config, s.char_size)
    poem = Poem(config)
    poem.train()
    print(poem.predict())
    poem.close()


