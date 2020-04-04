import tensorflow as tf
import numpy as np
import argparse
import os


class Config:
    def __init__(self):
        self.batch_size = 2
        self.num_step = 8 * 4
        self.num_units = 5#200
        self.gpus = self.get_gpus()
        self.classes = 4
        self.ch_size = 200

        self.name = 'p25'
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
        self.sub_tensors = []
        self.config = config
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)

        with tf.variable_scope('segmentation'):
            for gpu_index in range(config.gpus):
                self.sub_tensors.append(Sub_tensors(config, gpu_index, opt))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            grad = self.merge_grads()
            self.train_op = opt.apply_gradients(grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_tensors])
            tf.summary.scalar('loss', self.loss)
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
        result = [(tf.reduce_mean(grads[v], axis = 0),v) for v in grads]
        for v in indexed_grads:
            indices = tf.concat([g.indices for g in indexed_grads[v]], axis = 0)
            values = tf.concat([g.values for g in indexed_grads[v]], axis = 0)
            g = tf.IndexedSlices(values, indices)
            result.append((g, v))
        return result


class Sub_tensors:
    def __init__(self, config: Config, gpu_index, opt:tf.train.AdamOptimizer):
        with tf.device('/gpu:%d' % gpu_index):
            self.x = tf.placeholder(tf.int32, [None, config.num_step], 'x')
            char_dcit = tf.get_variable('char_size', [config.ch_size, config.num_units], tf.float32)
            x = tf.nn.embedding_lookup(char_dcit, self.x)#-1, 32, num_units

            self.y = tf.placeholder(tf.int32, [None, config.num_step], 'x')
            y_onehot = tf.one_hot(self.y, config.classes)

            loss_list, self.predict_list = self.bi_rnn(x, y_onehot, config)
            self.loss = tf.reduce_mean(loss_list)
            self.grad = opt.compute_gradients(self.loss)

    def bi_rnn(self, x, y, config: Config):
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name = 'lstm1')
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(config.num_units, name='lstm2')

        batch_size_tf = tf.shape(x)[0]
        state1 = cell1.zero_state(batch_size_tf, tf.float32)
        state2 = cell2.zero_state(batch_size_tf, tf.float32)

        y1_list = []
        y2_list = []
        for i in range(config.num_step):
            xi1 = x[:, i]
            yi1_pred, state1 = cell1(xi1, state1)
            y1_list.append(yi1_pred)

            xi2 = x[:, config.num_step - i - 1]
            yi2_pred, state2 = cell2(xi2, state2)
            y2_list.insert(0, yi2_pred)



        loss_list = []
        y_predict_list = []
        with tf.variable_scope('rnn'):
            for i in range(config.num_step):
                x = y1_list[i] + y2_list[i]
                y_predict = tf.layers.dense(x, config.classes, name = 'y_predict')
                tf.get_variable_scope().reuse_variables()

                y_predict_list.append(tf.math.argmax(y_predict, axis = 1, output_type=tf.int32))
                yi = y[:, i]

                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,labels=yi)
                loss_list.append(loss)
        return loss_list, y_predict_list





class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0
        self.data = list(np.random.randint(0, config.ch_size, [self.num(), config.num_step]))
        self.label = list(np.random.randint(0, config.classes, [self.num(), config.num_step]))

    def num(self):
        return 100

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.num():
            x = self.data[self.index: next]
            y = self.label[self.index: next]
        else:
            x = self.data[self.index:]
            y = self.label[self.index:]
            next -= self.num()
            x += self.data[:next]
            y += self.label[:next]
        self.index = next
        return x, y






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
            self.writer = tf.summary.FileWriter(config.logdir, self.session.graph)
            try:
                self.saver.restore(self.session, config.save_path)
                print('Restore the model from %s successfully.' % config.save_path)
            except:
                print('Fail to restore the model from', config.save_path)
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
                _, loss, su = self.session.run(
                    [self.tensors.train_op, self.tensors.loss, self.tensors.summary_op], feed_dict)
                self.writer.add_summary(su, epoch * batches + batch)
                print('%d/%d: loss=%.8f' % (batch, epoch, loss), flush=True)
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path, flush=True)

    def predict(self):
        pass

    def close(self):
        self.session.close()

if __name__ == '__main__':
    config = Config()

    app = App(config)
    app.train()
    app.close

    # a = np.array([1,2,3])
    # b = np.array([1])
    # print(a+b)
