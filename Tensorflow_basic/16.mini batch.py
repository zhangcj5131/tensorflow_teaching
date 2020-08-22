import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import pprint

# def batches(features, labels, batch_size):
#     assert len(features) == len(labels)
#     result_array = []
#     for start in range(0, len(features), batch_size):
#         end = start + batch_size
#         x = features[start: end]
#         y = labels[start: end]
#         result_array.append((x, y))
#     return result_array


def get_batches(features, labels, batch_size):
    assert len(features) == len(labels)
    for start in range(0, len(features), batch_size):
        end = start + batch_size
        x = features[start: end]
        y = labels[start: end]
        yield x, y

features = [[1,2,3,4],
            [5,6,7,8],
            [8,9,10,11],
            [12,13,14,15]]
labels = [[1,2],[3,4],[5,6],[7,8]]


class Config:
    def __init__(self):
        self.epoches = 50
        self.lr = 0.01
        self.name = 13
        self.logdir = './log/{name}/{name}'.format(name=self.name)
        self.save_path = './models/{name}/{name}'.format(name=self.name)
        self.sample_path = '../data/MNIST_data'
        self.batch_size = 128
        self.classes_num = 10
        self.features_num = 784


class Sample:
    def __init__(self, config: Config):
        self.config = config
        self.ds = read_data_sets(config.sample_path)

    def next_batch(self, batch_size):
        return self.ds.train.next_batch(batch_size)

    @property
    def num(self):
        return self.ds.train.num_examples

class Model:
    def __init__(self, config: Config):
        self.config = config
        with tf.Graph().as_default() as graph:
            gpu_options = tf.GPUOptions(
                allow_growth=True,  # 不预先分配整个gpu显存计算，而是从小到大，按需增加。
                per_process_gpu_memory_fraction=0.9  # value-(0, 1) 限制使用该gpu设备的显存使用的百分比。一般建议设置0.8--0.9左右。
            )
            conf = tf.ConfigProto(
                allow_soft_placement=True,
                # 是否允许tf动态的使用cpu和gpu。当我们的版本是gpu版本，那么tf会默认调用你的gpu:0来运算，如果你的gpu无法工作，那么tf就会报错。所以建议有gpu版本的同学，将这个参数设置为True。
                log_device_placement=False,  # bool值，是否打印设备位置的日志文件。
                gpu_options=gpu_options
            )
            self.session = tf.Session(graph=graph, config=conf)
            self.tensors = Tensors(config)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph=graph)
            self.saver = tf.train.Saver()
            try:
                # self.saver.restore(self.session, config.save_path)
                dirname = os.path.dirname(config.save_path)
                ckpt = tf.train.get_checkpoint_state(dirname)
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print('model was successfully restored!')
            except:
                print('the model does not exist, we have to train a new one!')
                self.session.run(tf.global_variables_initializer())

    def train(self):
        config = self.config
        samples = Sample(config)
        batches = samples.num // config.batch_size
        step = 1
        for epoch in range(config.epoches):
            for batch in range(batches):
                x, y = samples.next_batch(config.batch_size)
                feed_dict = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                    self.tensors.lr: config.lr
                }
                _, su, lo, acc = self.session.run([self.tensors.train_op,
                                  self.tensors.summary_op,
                                  self.tensors.loss,
                                  self.tensors.accuracy], feed_dict=feed_dict)
                self.file_writer.add_summary(su, step)
                step += 1
            if epoch % 10 == 0:
                print('epoch=%d, loss=%f, acc=%f' % (epoch, lo, acc))
                self.saver.save(self.session, config.save_path, global_step=epoch)

    def close(self):
        self.file_writer.close()
        self.session.close()





class Tensors:
    def __init__(self, config: Config):
        self.config = config
        with tf.device('/gpu:0'):
            with tf.variable_scope('network'):
                self.x = tf.placeholder(tf.float32, [None, config.features_num], 'x')
                self.y = tf.placeholder(tf.int32, [None], 'y')
                y = tf.one_hot(self.y, config.classes_num)
                self.lr = tf.placeholder(tf.float32, [], 'lr')

                w = tf.get_variable('w', [config.features_num, config.classes_num],
                                    tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [config.classes_num],
                                    tf.float32, initializer=tf.zeros_initializer())
                logits = tf.matmul(self.x, w) + b
            with tf.variable_scope('loss'):
                softmax_value = tf.nn.softmax(logits, axis = 1)
                self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(softmax_value) * y))
                tf.summary.scalar('loss', self.loss)

            with tf.variable_scope('optimizer'):
                opt = tf.train.GradientDescentOptimizer(self.lr)
                self.train_op = opt.minimize(self.loss)

            with tf.variable_scope('accuracy'):
                self.y_pred = tf.math.argmax(logits, axis = 1, output_type=tf.int32)
                correct = tf.equal(self.y_pred, self.y)
                correct = tf.cast(correct, tf.float32)
                self.accuracy = tf.reduce_mean(correct)
                tf.summary.scalar('accuracy', self.accuracy)
                self.summary_op = tf.summary.merge_all()

if __name__ == '__main__':
    config = Config()
    model = Model(config)
    model.train()
    # s = Sample(config)
    # print(s.num())

