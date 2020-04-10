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
        self.sub_tensors = []
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.int32, [], 'lr')
            opt = tf.train.AdadeltaOptimizer(self.lr)

        with tf.variable_scope('dialog'):
            for gpu_index in range(config.gpus):
                self.sub_tensors.append(Sub_tensors(gpu_index, config, opt))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grad = self.merge_grads()
            self.train_op = opt.apply_gradients(grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_tensors])
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def merge_grads(self):
        indices_grads = {}
        grads = {}
        for ts in self.sub_tensors:
            for g, v in ts.grad:
                if isinstance(g, tf.IndexedSlices):
                    if v not in indices_grads:
                        indices_grads[v] = []
                    indices_grads[v].append(g)
                else:
                    if v not in grads:
                        grads[v] = []
                    grads[v].append(g)
        results = [(tf.reduce_mean(grads[v], axis = 0),v) for v in grads]
        for v in indices_grads:
            indices = tf.concat([g.indices for g in indices_grads[v]], axis = 0)
            values = tf.concat([g.values for g in indices_grads[v]], axis=0)
            g = tf.IndexedSlices(values, indices)
            results.append((g, v))
        return results

class Sub_tensors:
    def __init__(self, gpu_index, config: Config, opt: tf.train.AdadeltaOptimizer):
        self.config = config
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.int32, [None, config.num_step1], 'x')
            self.y = tf.placeholder(tf.int32, [None, config.num_step2], 'y')
            self.z = tf.placeholder(tf.int32, [None, config.num_step2], 'z')

            #step1, -1, units
            out_list1, reading_state = self.encode(self.x, None, config.num_step1, 'reading')
            #step2, -1, units
            out_list2, question_state = self.encode(self.y, reading_state, config.num_step2, 'question')

            #step1+step2, -1, units
            out_list = out_list1 + out_list2
            #-1, step1, step2
            out_list = tf.transpose(out_list, [1, 0, 2])

            losses, self.y_predict = self.answer(self.z, question_state, out_list, config.num_step2, 'answer')
            self.loss = tf.reduce_mean(losses)
            self.grad = opt.compute_gradients(self.loss)

    def answer(self, z, state, out_list, step, name):
        config = self.config
        with tf.variable_scope(name):
            z = tf.one_hot(z, config.ch_size)

            cell = self.get_cell()

            losses = []
            y_predict_list = []
            for i in range(step):
                xi = self.get_soft_attention(state, out_list)
                y_pred, state = cell(xi, state)
                y_pred = tf.layers.dense(y_pred, config.ch_size, name = 'y_predict')
                tf.get_variable_scope().reuse_variables()

                zi = z[:, i]

                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=zi)
                losses.append(loss)
                y_predict_list.append(tf.math.argmax(y_pred, axis = 1, output_type=tf.int32))
        return losses, y_predict_list

    #out_list:-1, step1+step2, units
    #state: [(c, h), (c, h)]
    def get_soft_attention(self, state, out_list):
        config = self.config
        total_step = config.num_step1 + config.num_step2
        #state:-1, 4*units
        state = self.convert_state(state)
        alpha = tf.reshape(out_list, [-1, total_step * config.num_units])
        alpha = tf.concat([alpha, state], axis = 1)#-1, (step1+step2)*units + 4*units
        with tf.variable_scope('attention'):
            alpha = tf.layers.dense(alpha, total_step, activation=tf.nn.relu, name = 'dense1')
            alpha = tf.layers.dense(alpha, total_step, name='dense2')#-1, step1+step2
            alpha = tf.reshape(alpha, [-1, total_step, 1])
            alpha = tf.nn.softmax(alpha, axis = 1)#-1, step1+step2, 1
            attention = alpha * out_list#-1, step1+step2, units
            attention = tf.reduce_sum(attention, axis = 1)#-1, units
            return attention

    def convert_state(self, state):
        config = self.config
        result = []
        for s in state:
            result.append(s.c)
            result.append(s.h)
        #result:4, -1, units
        # result:-1, 4, units
        result = tf.transpose(result, [1, 0, 2])
        result = tf.reshape(result, [-1, config.levels*2*config.num_units])
        return result





    def encode(self, x, state, step, name):
        config = self.config
        with tf.variable_scope(name):
            char_dict = tf.get_variable('char_dict', [config.ch_size, config.num_units], tf.float32)
            x = tf.nn.embedding_lookup(char_dict, x) #[-1, step1, units]

            cell = self.get_cell()

            if state is None:
                batch_size = tf.shape(x)[0]
                state = cell.zero_state(batch_size, tf.float32)
            out_list = []
            for i in range(step):
                xi = x[:, i]
                outi, state = cell(xi, state)
                out_list.append(outi)
            return out_list, state

    def get_cell(self):
        cells = []
        for i in range(self.config.levels):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_units, name = 'lstm_%d' % i)
            cells.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells)



class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0
        result = []
        record_num = 0
        for i in range(self.num()):
            reading = np.random.randint(0, config.ch_size, [config.num_step2*2])
            dialog_num = np.random.randint(2, 20)
            record_num += 2*dialog_num
            dialogs = []
            for _ in range(dialog_num):
                who = np.random.randint(0, 2)
                is_question = np.random.randint(0, 2)
                what = np.random.randint(0, config.ch_size, [config.num_step2])
                dialogs.append((who, is_question, what))
            result.append((reading, dialogs))
        self.data = result
        self.record_num = record_num
        self.samples = self.get_samples()

    def get_samples(self):
        while True:
            reading, dialogs = self.data[self.index]
            self.index = (self.index + 1) % self.num()
            for x, y, z in self.get_sample(reading, dialogs, 0):
                yield x, y, z
            for x, y, z in self.get_sample(reading, dialogs, 1):
                yield x, y, z

    def get_sample(self, reading, dialogs, questioner):
        config = self.config
        q = None
        for index, (who, is_question, what) in enumerate(dialogs):

            if who == questioner:
                if is_question:
                    q = what
                else:
                    reading = self.add2reading(reading, what, who)
            else:
                if is_question:
                    q = np.zeros([config.num_step2])
                    yield self.padding(reading, config.num_step1), q, what
                else:
                    if q is not None:
                        yield self.padding(reading, config.num_step1), q, what
                        q = None

    def padding(self, reading, max_length):
        length = len(reading)
        if length > max_length:
            return reading[:max_length]
        reading = np.concatenate([reading, [0] * (max_length - length)], axis = 0)
        return reading


    def add2reading(self, reading, what, who):
        what = np.insert(what, 0, who)
        reading = np.concatenate([reading, what], axis = 0)
        return reading





    def next_batch(self, batch_size):
        xs, ys, zs = [], [], []
        for _ in range(batch_size):
            x, y, z = next(self.samples)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return xs, ys, zs







    def num(self):
        return 100





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
            batches = self.samples.record_num // (cfg.gpus * cfg.batch_size)
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: cfg.lr
                }

                for gpu_index in range(cfg.gpus):
                    xr, xq, y = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = xr
                    feed_dict[self.tensors.sub_tensors[gpu_index].y] = y
                    feed_dict[self.tensors.sub_tensors[gpu_index].z] = xq

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
B 的提问,把 q=0, 得到提问
B 的陈述,如果是对 A的回答,
B 的陈述,不是回答,则忽略
'''
if __name__ == '__main__':
    config = Config()
    # config.from_cmd_line()

    # s = Samples(config)
    # x, y, z = s.next_batch(20)
    # print(y)
    # print(s.record_num)

    app = App(config)
    app.train()
    app.close()
    print('Finished!')
