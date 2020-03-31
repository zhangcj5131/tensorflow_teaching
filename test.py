import tensorflow as tf
import numpy as np
import os
import argparse


class Config:
    def __init__(self):
        self.batch_size = 20
        self.num_step1 = 20
        self.num_step2 = 10
        self.num_step3 = 10
        self.num_units = 5
        self.levels = 2

        self.gpus = self.get_gpus()
        self.ch_size = 200

        self.name = 'p28'
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
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []

        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('poem'):
            for gpu_index in range(config.gpus):
                self.sub_ts.append(SubTensors(config, gpu_index, opt))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grad = self.merge_grads()

            self.train_op = opt.apply_gradients(grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_ts])

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

        total = 0
        for var in tf.trainable_variables():
            num = self.get_param_num(var.shape)
            print(var.name, var.shape, num)
            total += num
        print('Total params:', total)

    def get_param_num(self, shape):
        num = 1
        for sh in shape:
            num *= sh.value
        return num

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
    def __init__(self, config, gpu_index, opt):
        self.config = config
        with tf.device('/gpu:%d' % gpu_index):
            self.xr = tf.placeholder(tf.int32, [None, config.num_step1], 'xr')
            self.xq = tf.placeholder(tf.int32, [None, config.num_step2], 'xq')
            self.y = tf.placeholder(tf.int32, [None, config.num_step3], 'y')

            reading_vector = self.call_cell(self.xr, None, config.num_step1, 'reading')
            question_vector = self.call_cell(self.xq, reading_vector, config.num_step2, 'question')

            losses, self.y_predict = self.answer(question_vector, self. y, 'answer')

            self.loss = tf.reduce_mean(losses)
            self.grad = opt.compute_gradients(self.loss)

    def call_cell(self, x, state, num_step, name):
        config = self.config
        with tf.variable_scope(name):
            cell = self.new_cell()
            cell1 = self.new_cell()
            if state is None:
                batch_size_ts = tf.shape(x)[0]
                state = cell.zero_state(batch_size_ts, tf.float32)

            ch_dict = tf.get_variable('ch_dict', [config.ch_size, config.num_units], tf.float32)
            x = tf.nn.embedding_lookup(ch_dict, x)  # [-1, num_step, num_units]
            for i in range(num_step):
                xi = x[:, i, :]  # [-1, num_units]
                _, state = cell(xi, state)

                #tf.get_variable_scope().reuse_variables()

            return state

    def new_cell(self):
        cells = []
        for i in range(self.config.levels):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_units, name='lstm_%d' % i)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell

    def answer(self, vector, y, name):
        #  y: [-1, num_step3]
        y = tf.one_hot(y, self.config.ch_size)  # [-1, num_step3, ch_size]
        with tf.variable_scope(name):
            cell = self.new_cell()
            xi = tf.zeros([tf.shape(y)[0], self.config.num_units])

            losses = []
            y_predict = []
            for i in range(self.config.num_step3):
                yi_predict, vector = cell(xi, vector)  # [-1, num_units], ...

                yi = y[:, i, :]  # [-1, ch_size]
                yi_predict = tf.layers.dense(yi_predict, self.config.ch_size, use_bias=False, name='dense')  # [-1, ch_size]
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yi, logits=yi_predict)
                losses.append(loss)
                y_predict.append(tf.argmax(yi_predict, axis=1))
                tf.get_variable_scope().reuse_variables()

        return losses, y_predict






    def encode_decode(self, x, y, config):
        with tf.variable_scope('encode'):
            vector, attention = self.encode(x, config)

        with tf.variable_scope('decode'):
            losses, y_predict = self.decode(vector, attention, config)

        return losses, y_predict

    def encode(self, x, config):
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='encoder_lstm1')
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='encoder_lstm2')
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        ch_dict = tf.get_variable('ch_dict', [config.ch_size, config.num_units], tf.float32)
        x = tf.nn.embedding_lookup(ch_dict, x)  # [-1, num_step1, num_units]

        batch_size_ts = tf.shape(x)[0]
        state = cell.zero_state(batch_size_ts, tf.float32)
        z = []
        with tf.variable_scope('rnn'):
            for i in range(config.num_step1):
                xi = x[:, i, :]  # [-1, num_units]
                zi, state = cell(xi, state)
                z.append(zi)
                tf.get_variable_scope().reuse_variables()

            # z: [num_step1, -1, num_units]
        attention = self.get_attention(z, config)
        return state, attention  # [-1, num_units], [-1, num_step2, num_units]

    def get_attention(self, z, config):
        # z: [num_step1, -1, num_units]
        with tf.variable_scope('attention'):
            z = tf.transpose(z, [1, 0, 2])  # [-1, num_step1, num_units]

            t = z  # [-1, num_step1, num_units]
            t = tf.reshape(t, [-1, config.num_step1 * config.num_units])
            t = tf.layers.dense(t, config.num_step2 * config.num_step1, name='dense')  # [-1, num_step2 * num_step1]
            t = tf.reshape(t, [-1, config.num_step2, config.num_step1, 1])
            t = tf.nn.softmax(t, axis=2)  # [-1, num_step2, num_step1, 1]

            z = tf.reshape(z, [-1, 1, config.num_step1, config.num_units])
            z = z * t  # [-1, num_step2, num_step1, num_units]
            attention = tf.reduce_sum(z, axis=2)  # [-1, num_step2, num_units]

            # attention: [-1, num_step2, num_units]
            return attention

    def decode(self, state, attention, config):
        # state: [-1, num_units]
        # attention: [-1, num_step2, num_units]
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='decoder_lstm1')
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='decoder_lstm2')
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        with tf.variable_scope('rnn'):
            y = tf.one_hot(self.y, config.en_size)  # [-1, num_step2, en_size]

            y_predict = []
            losses = []
            for i in range(config.num_step2):
                xi = attention[:, i, :]  # [-1, num_units]
                yi_predict, state = cell(xi, state)  # [-1, num_units]
                yi = y[:, i, :]  # [-1, en_size]
                y_predict.append(tf.argmax(yi_predict, axis=1))

                logits = tf.layers.dense(yi_predict, config.en_size, name='dense')  # [-1, en_size]
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yi, logits=logits)
                losses.append(loss)

                tf.get_variable_scope().reuse_variables()

        return losses, y_predict


class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0

        result = []
        for _ in range(self.num()):
            reading = np.random.randint(0, config.ch_size, [config.num_step1])
            question_num = np.random.randint(1, 20)
            q_s = np.random.randint(0, config.ch_size, [question_num, config.num_step2])
            a_s = np.random.randint(0, config.ch_size, [question_num, config.num_step3])
            result.append((reading, q_s, a_s))
        self.data = result

    def num(self):
        """
        The number of the samples in total.
        :return:
        """
        return 100

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.num():
            result = self.data[self.index: next]
        else:
            result = self.data[self.index:]
            next -= self.num()
            result += self.data[:next]
        self.index = next
        return result  # [batch_size] ==> ([num_step1], [?, num_step2], [?, num_step3])


class QASamples:
    def __init__(self, big_samples):
        self.big_samples = big_samples
        self.big_index = 0
        self.small_index = 0
        self.stop = False
        self.stop_yield = False
        self.small_samples = self.new_small_samples()

    def new_small_samples(self):
        while not self.stop_yield:
            big_sample = self.big_samples[self.big_index]

            xr = big_sample[0]
            xq = big_sample[1][self.small_index]
            y = big_sample[2][self.small_index]

            yield xr, xq, y

            self.small_index += 1
            if self.small_index >= len(big_sample[1]):
                self.small_index = 0
                self.big_index += 1
                if self.big_index >= len(self.big_samples):
                    self.big_index = 0
                    self.stop = True
        yield None, None, None

    def next_batch(self, batch_size):
        if self.stop:
            self.stop_yield = True
            self.get_small_sample()
            return None, None, None

        xas = []
        xqs = []
        ys = []
        for _ in range(batch_size):
            xa, xq, y = self.get_small_sample()  # rv: [(c, h), (c, h)]
            xas.append(xa)
            xqs.append(xq)
            ys.append(y)
        # rvs: [batch_size] --> [(c, h), (c, h)]

        return xas, xqs, ys

    def get_small_sample(self):
        return next(self.small_samples)


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.samples = Samples(config)
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

    # def get_reading_vectors(self, big_samples):
    #     batch = [s[0] for s in big_samples]
    #     feed_dict = {
    #         self.tensors.lr: self.config.lr
    #     }
    #     feed_dict[self.tensors.sub_ts[0].xr] = batch
    #     rvs = self.session.run(self.tensors.sub_ts[0].reading_vector, feed_dict)
    #     return rvs

    def train(self):
        cfg = self.config
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)

        for epoch in range(cfg.epoches):
            batches = self.samples.num() // (cfg.gpus * cfg.batch_size)
            for batch in range(batches):
                samples = self.samples.next_batch(cfg.batch_size)  # [batch_size] --> ([n_s1], [?, n_s2], [?, n_s3])
                qa_samples = QASamples(samples)

                while True:
                    xas, xqs, ys = qa_samples.next_batch(cfg.batch_size * cfg.gpus)
                    if xas is None:
                        break

                    feed_dict = {
                        self.tensors.lr: cfg.lr
                    }

                    start = 0
                    for gpu_index in range(cfg.gpus):

                        end = start + cfg.batch_size
                        feed_dict[self.tensors.sub_ts[gpu_index].xq] = xqs[start: end]
                        feed_dict[self.tensors.sub_ts[gpu_index].y] = ys[start: end]
                        feed_dict[self.tensors.sub_ts[gpu_index].xr] = xas[start: end]
                        start = end

                    _, loss, su = self.session.run(
                        [self.tensors.train_op, self.tensors.loss, self.tensors.summary_op], feed_dict)
                    writer.add_summary(su, epoch * batches + batch)
                    print('%d/%d: loss=%.8f' % (batch, epoch, loss), flush=True)
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path, flush=True)

    def predict(self):
        pass

    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()

    # config.gpus=2
    # Tensors(config)

    app = App(config)

    app.train()
    # app.close()
    print('Finished!')
