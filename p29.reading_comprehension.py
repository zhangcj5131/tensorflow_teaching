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
        self.sub_tensors = []
        with tf.device('/gpu:0'):
            self.lr = tf.placeholder(tf.int32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope('reading'):
            for gpu_index in range(config.gpus):
                self.sub_tensors.append(Sub_tensors(gpu_index, config, opt))
                tf.get_variable_scope().reuse_variables()
        with tf.device('/gpu:0'):
            self.grad = self.merge_grads()
            self.train_op = opt.apply_gradients(self.grad)
            self.loss = tf.reduce_mean([ts.loss for ts in self.sub_tensors])
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            self.show_para()

    def show_para(self):
        total = 0
        for var in tf.trainable_variables():
            num = self.get_var_num(var.shape)
            print(var.name, var.shape, num)
            total += num
        print('total para num is', total)

    def get_var_num(self, shape):
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
        result = [(tf.reduce_mean(grads[v], axis = 0),v) for v in grads]
        for v in indexed_grads:
            indices = tf.concat([g.indices for g in indexed_grads[v]], axis = 0)
            values = tf.concat([g.values for g in indexed_grads[v]], axis=0)
            g = tf.IndexedSlices(values, indices)
            result.append((g, v))
        return result

class Sub_tensors:
    def __init__(self, gpu_index, config: Config, opt: tf.train.AdamOptimizer):
        self.config = config
        with tf.device('/gpu:%d' % gpu_index):
            self.articles = tf.placeholder(tf.int32, [None, config.num_step1], 'articles')
            self.questions = tf.placeholder(tf.int32, [None, config.num_step2], 'questions')
            self.answers = tf.placeholder(tf.int32, [None, config.num_step3], 'answers')

            article_state = self.get_state(None, self.articles, config.num_step1, 'articles')
            question_state = self.get_state(article_state, self.questions, config.num_step2, 'questions')

            loss_list, self.y_predict = self.get_answer(question_state, self.answers, config.num_step3, 'answer')

            self.loss = tf.reduce_mean(loss_list)
            self.grad = opt.compute_gradients(self.loss)


    def get_answer(self, state, y, num_step, name):
        with tf.variable_scope(name):
            y = tf.one_hot(y, self.config.ch_size)

            cell = self.get_cell()

            batch_size = tf.shape(y)[0]
            xi = tf.zeros([batch_size, config.num_units], tf.float32)

            loss_list = []
            y_list = []
            for i in range(num_step):
                y_pred, state = cell(xi, state)
                y_pred = tf.layers.dense(y_pred, self.config.ch_size, name = 'y_predict')
                tf.get_variable_scope().reuse_variables()

                yi = y[:, i]
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=yi)
                loss_list.append(loss)

                y_index = tf.math.argmax(y_pred, axis = 1, output_type=tf.int32)
                y_list.append(y_index)
        return loss_list, y_list



    def get_state(self, state, x, num_step, name):
        with tf.variable_scope(name):
            cell = self.get_cell()

            char_dict = tf.get_variable('char_dict', [config.ch_size, config.num_units], tf.float32)
            x = tf.nn.embedding_lookup(char_dict, x)#-1, num_step, num_units

            if state is None:
                batch_size = tf.shape(x)[0]
                state = cell.zero_state(batch_size, tf.float32)
            for i in range(num_step):
                xi = x[:, i]
                _, state = cell(xi, state)

        return state

    def get_cell(self):
        cells = []
        for i in range(config.levels):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_units, name = 'lstm_%d' % i)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell


class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0
        self.articles = []
        self.questions = []
        self.answers = []
        for i in range(self.num()):
            article = np.random.randint(0, config.ch_size, [config.num_step1])
            question_num = np.random.randint(1, 20)
            for j in range(question_num):
                question = np.random.randint(0, config.ch_size, [config.num_step2])
                answer = np.random.randint(0, config.ch_size, [config.num_step3])
                self.articles.append(article)
                self.questions.append(question)
                self.answers.append(answer)

    def num(self):
        return 100

    def record_num(self):
        return len(self.answers)

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.record_num():
            articles = self.articles[self.index: next]
            questions = self.questions[self.index: next]
            answers = self.answers[self.index: next]
        else:
            articles = self.articles[self.index: ]
            questions = self.questions[self.index: ]
            answers = self.answers[self.index: ]
            next -= self.record_num()
            articles += self.articles[:next]
            questions += self.questions[:next]
            answers += self.answers[:next]
        self.index = next
        return articles, questions, answers





class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config = conf, graph=graph)
            self.samples = Samples(config)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph=graph)
            self.tensors = Tensors(config)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, config.save_path)
            except:
                self.session.run(tf.global_variables_initializer())

    def train(self):
        config = self.config
        batches = self.samples.record_num() // (config.batch_size * config.gpus)
        for epoch in range(config.epoches):
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: config.lr
                }
                for gpu_index in range(config.gpus):
                    articles, questions, answers = self.samples.next_batch(config.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].articles] = articles
                    feed_dict[self.tensors.sub_tensors[gpu_index].questions] = questions
                    feed_dict[self.tensors.sub_tensors[gpu_index].answers] = answers

                _, loss, su = self.session.run([self.tensors.train_op,
                                                self.tensors.loss,
                                                self.tensors.summary_op], feed_dict)
                self.file_writer.add_summary(su, epoch * batches + batch)
                print('%d/%d loss = %f' % (epoch, batch, loss))
            self.saver.save(self.session, config.save_path)

    def close(self):
        self.session.close()















if __name__ == '__main__':
    config = Config()
    # s = Samples(config)
    #
    # print(len(s.next_batch(124)[1]))

    app = App(config)

    app.train()
    app.close()
    print('Finished!')
