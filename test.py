import tensorflow as tf
import numpy as np
import os
import argparse


class Config:
    def __init__(self):
        self.batch_size = 20
        self.num_step1 = 50  # the length of the background
        self.num_step2 = 10  # the length of the question and answer
        self.num_units = 5
        self.levels = 2

        self.gpus = self.get_gpus()
        self.ch_size = 200

        self.name = 'p30'
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
            self.y = tf.placeholder(tf.int32, [None, config.num_step2], 'y')

            background_vector = self.call_cell(self.xr, None, config.num_step1, 'background')
            question_vector = self.call_cell(self.xq, background_vector, config.num_step2, 'question')

            losses, self.y_predict = self.answer(question_vector, self.y, 'answer')

            self.loss = tf.reduce_mean(losses)
            self.grad = opt.compute_gradients(self.loss)

    def call_cell(self, x, state, num_step, name):
        config = self.config
        with tf.variable_scope(name):
            cell = self.new_cell()

            if state is None:
                batch_size_ts = tf.shape(x)[0]
                state = cell.zero_state(batch_size_ts, tf.float32)

            ch_dict = tf.get_variable('ch_dict', [config.ch_size, config.num_units], tf.float32)
            x = tf.nn.embedding_lookup(ch_dict, x)  # [-1, num_step, num_units]
            for i in range(num_step):
                xi = x[:, i, :]  # [-1, num_units]
                _, state = cell(xi, state)
                tf.get_variable_scope().reuse_variables()

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
            for i in range(self.config.num_step2):
                yi_predict, vector = cell(xi, vector)  # [-1, num_units], ...

                yi = y[:, i, :]  # [-1, ch_size]
                yi_predict = tf.layers.dense(yi_predict, self.config.ch_size, use_bias=False, name='dense')  # [-1, ch_size]
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yi, logits=yi_predict)
                losses.append(loss)
                y_predict.append(tf.argmax(yi_predict, axis=1))
                tf.get_variable_scope().reuse_variables()

        return losses, y_predict


class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0

        result = []
        for _ in range(self.num()):
            reading = np.random.randint(0, config.ch_size, [np.random.randint(0, 2*config.num_step2)])
            dialog_num = np.random.randint(1, 20)

            dialogs = []
            for _ in range(dialog_num):
                #谁在说话
                who = np.random.randint(0, 2)
                #是提问还是回答
                question = np.random.randint(0, 2)  # 1 means question, 0 means answer
                #内容
                what = np.random.randint(0, config.ch_size, [config.num_step2])
                dialogs.append((who, question, what))

            result.append((reading, dialogs))
        self.data = result
        self.samples = self.get_samples()

    def num(self):
        """
        The number of the samples in total.
        :return:
        """
        return 100

    def next_batch(self, batch_size):
        xrs, xqs, ys = [], [], []
        for _ in range(batch_size):
            xr, xq, y = next(self.samples)
            xrs.append(xr)
            xqs.append(xq)
            ys.append(y)
        #  xrs = [-1, num_step1], xqs = [-1, num_step2], ys = [-1, num_step2]
        return xrs, xqs, ys

    def get_samples(self):
        while True:
            reading, dialogs = self.data[self.index]
            # ****** add *******
            self.index = (self.index + 1) % len(self.data)
            # ******************
            #0为提问,1 是回答
            for xr, xq, y in self.get_sample(reading, dialogs, 0):
                yield xr, xq, y
            # 1为提问,0 是回答
            for xr, xq, y in self.get_sample(reading, dialogs, 1):
                yield xr, xq, y

    def get_sample(self, reading, dialogs, questioner):
        q = None
        for index, (who, question, what) in enumerate(dialogs):
            if who == questioner:
                if question:
                    q = what
                else:
                    for word_id in self.transform2background(what, who):
                        reading = np.concatenate((reading, [word_id]), axis=0)
            else:
                if question:
                    zero = np.zeros([self.config.num_step2])
                    yield self.padding(reading, self.config.num_step1), zero, what
                else:
                    if q is not None:
                        yield self.padding(reading, self.config.num_step1), q, what
                        q = None

    def padding(self, reading, max_length):
        length = len(reading)
        if max_length < length:
            print('The max_length (%d) is not enought for reading' % max_length, flush=True)
            return reading[0:max_length]

        reading = np.concatenate((reading, [0] * (max_length - length)), axis=0)
        return reading

    def transform2background(self, what, who):
        return what


class App:
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
        self.samples = Samples(cfg)

        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)

        for epoch in range(cfg.epoches):
            batches = self.samples.num() // (cfg.gpus * cfg.batch_size)
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: cfg.lr
                }

                for gpu_index in range(cfg.gpus):
                    xr, xq, y = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_ts[gpu_index].xr] = xr
                    feed_dict[self.tensors.sub_ts[gpu_index].y] = y
                    feed_dict[self.tensors.sub_ts[gpu_index].xq] = xq

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

'''
一个数据两个人 AB分饰两个角色,用户和机器人,做两轮训练
提问者是人,回答者是机器人,提问者的陈述句被加入背景

对于训练数据,由于双方会交换,所以所有的背景描述都需要客观,不能有偏向.
对于预测数据,在明确谁是酒店谁是客人以后,在背景中以酒店为对象,加入客户的陈述的时候要说明是来自客户的
A问 B 答
A:是人
B:机器人
A的提问,全部当做 question
A的陈述加入 background
B 的提问,全部作为回答
'''
if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()

    # config.gpus=2
    # Tensors(config)

    app = App(config)

    app.train()
    app.close()
    print('Finished!')
