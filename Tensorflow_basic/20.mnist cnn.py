import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets




class Config:
    def __init__(self):
        self.name = 20
        self.epoches = 10
        self.batch_size = 128
        self.logdir_train = './log/{name}/train'.format(name = self.name)
        self.logdir_valid = './log/{name}/valid'.format(name=self.name)
        self.save_path = './models/{name}/{name}'.format(name = self.name)
        self.sample_path = './data/MNIST_data'
        self.keep_prob = 0.8
        self.classes = 10
        self.filter_size = 32
        self.lr = 0.01

class Samples:
    def __init__(self, config: Config):
        self.ds = read_data_sets(config.sample_path)

    def next_batch(self, batch_size):
        return self.ds.train.next_batch(batch_size)

    @property
    def num(self):
        return self.ds.train.num_examples

"""
网络结构图。
1、input [N, 28, 28, 1]    N 代表批量
2、卷积1  卷积核[5, 5, 1, 32]  ---> 4-d tensor [-1, 28, 28, 32]
3、池化1  strides=2            ---> 4-d tensor [-1, 14, 14, 32]
4、卷积2  卷积核[5, 5, 32, 64]  --->            [-1, 14, 14, 64]
5、池化2  strides=2             --->            [-1, 7, 7, 64]
6、拉平层（reshape）            --->             [-1, 7*7*64]
7、FC1    权重[7*7*64, 1024]    --->             [-1, 1024]
8、输出层(logits)  权重[1024, num_classes]   --->   [-1, num_classes]
"""

class Tensors:
    def __init__(self, config: Config):
        self.config = config
        with tf.device('/gpu:0'):

            with tf.variable_scope('network'):
                weight_list = [
                    tf.get_variable('conv1_w', [5, 5, 1, 32], tf.float32, tf.truncated_normal_initializer(stddev=0.1)),
                    tf.get_variable('conv2_w', [5, 5, 32, 64], tf.float32, tf.truncated_normal_initializer(stddev=0.1)),
                    tf.get_variable('fc_w', [7*7*64, 1024], tf.float32, tf.truncated_normal_initializer(stddev=0.1)),
                    tf.get_variable('logits_w', [1024, config.classes], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
                ]

                bias_list = [
                    tf.get_variable('conv1_b', [32], tf.float32, tf.zeros_initializer()),
                    tf.get_variable('conv2_b', [64], tf.float32, tf.zeros_initializer()),
                    tf.get_variable('fc_b', [1024], tf.float32, tf.zeros_initializer()),
                    tf.get_variable('logits_b', [config.classes], tf.float32,
                                    tf.zeros_initializer())
                ]
                self.x = tf.placeholder(tf.float32, [None, 784], 'x')
                x = tf.reshape(self.x, [-1, 28, 28, 1])
                self.y = tf.placeholder(tf.int32, [None], 'y')
                y_onehot = tf.one_hot(self.y, config.classes)
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')

                conv1 = self.conv2d_block(x, weight_list[0], bias_list[0])
                conv1 = tf.nn.dropout(conv1, keep_prob=self.keep_prob)
                conv1 = self.maxpool(conv1)

                conv2 = self.conv2d_block(conv1, weight_list[1], bias_list[1])
                conv2 = tf.nn.dropout(conv2, keep_prob=self.keep_prob)
                conv2 = self.maxpool(conv2)

                shape = conv2.get_shape()
                fc_input = tf.reshape(conv2, [-1, shape[1]*shape[2]*shape[3]])
                fc = tf.matmul(fc_input, weight_list[2]) + bias_list[2]
                fc = tf.nn.relu6(fc)
                fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)

                logits = tf.matmul(fc, weight_list[3]) + bias_list[3]

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=y_onehot
                ))
                tf.summary.scalar('train_loss', self.loss, collections=['train'])
                tf.summary.scalar('valid_loss', self.loss, collections=['valid'])
                opt = tf.train.GradientDescentOptimizer(self.lr)
                self.train_op = opt.minimize(self.loss)

            with tf.variable_scope('accuracy'):
                correct_num = tf.equal(tf.math.argmax(logits, axis=1, output_type=tf.int32), self.y)
                correct_num = tf.cast(correct_num, tf.float32)
                self.acc = tf.reduce_mean(correct_num)
                tf.summary.scalar('train_acc', self.acc, collections=['train'])
                tf.summary.scalar('valid_acc', self.acc, collections=['valid'])

                self.summary_train = tf.summary.merge_all('train')
                self.summary_valid = tf.summary.merge_all('valid')


    def conv2d_block(self, input_tensor, filter_w, filter_b, stride=1, padding='SAME'):
        conv = tf.nn.conv2d(input_tensor, filter_w, [1, stride, stride, 1], padding)
        conv = tf.nn.bias_add(conv, filter_b)
        conv = tf.nn.relu6(conv)
        return conv

    def maxpool(self, input_tensor, k = 2):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        return tf.nn.max_pool(input_tensor, ksize, strides, 'SAME')


class Model:
    def __init__(self, config: Config):
        self.config = config
        with tf.Graph().as_default() as graph:
            gpu_options = tf.GPUOptions(
                allow_growth=True,  # 不预先分配整个gpu显存计算，而是从小到大，按需增加。
                per_process_gpu_memory_fraction=0.9  # value-(0, 1) 限制使用该gpu设备的显存使用的百分比。一般建议设置0.8--0.9左右。
            )
            con = tf.ConfigProto(
                allow_soft_placement=True,
                # 是否允许tf动态的使用cpu和gpu。当我们的版本是gpu版本，那么tf会默认调用你的gpu:0来运算，如果你的gpu无法工作，那么tf就会报错。所以建议有gpu版本的同学，将这个参数设置为True。
                log_device_placement=True,  # bool值，是否打印设备位置的日志文件。
                gpu_options=gpu_options
            )
            self.session = tf.Session(graph=graph, config = con)
            self.tensors = Tensors(config)
            self.saver = tf.train.Saver()
            self.filewriter_train = tf.summary.FileWriter(logdir=config.logdir_train, graph=graph)
            self.filewriter_valid = tf.summary.FileWriter(logdir=config.logdir_valid, graph=graph)
            ckpt = tf.train.get_checkpoint_state(config.save_path)
            try:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print('model is successfully resotred!')
            except:
                self.session.run(tf.global_variables_initializer())
                print('the model does not exist, we have to train a new one')

    def train(self):
        config = self.config
        samples = Samples(config)
        batches = samples.num // config.batch_size
        step = 0
        for epoch in range(config.epoches):
            for batch in range(batches):
                step += 1
                x, y = samples.next_batch(config.batch_size)
                feed_dict_train = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                    self.tensors.lr: config.lr,
                    self.tensors.keep_prob: config.keep_prob
                }
                self.session.run(self.tensors.train_op, feed_dict_train)
                if step % 20 ==0:
                    train_su, train_loss, train_acc = self.session.run([self.tensors.summary_train,
                                      self.tensors.loss,
                                      self.tensors.acc], feed_dict=feed_dict_train)
                    self.filewriter_train.add_summary(train_su, global_step=step)

                    feed_dict_valid = {
                        self.tensors.x: samples.ds.validation.images[:512],
                        self.tensors.y: samples.ds.validation.labels[:512],
                        self.tensors.keep_prob: 1.
                    }
                    valid_su, valid_loss, valid_acc = self.session.run([self.tensors.summary_train,
                                                                        self.tensors.loss,
                                                                        self.tensors.acc], feed_dict=feed_dict_valid)
                    self.filewriter_valid.add_summary(valid_su, global_step=step)
                    print('epoch=%d, step=%d, train loss=%f, train acc=%f, valid loss=%f, valid acc=%f' %
                          (epoch, step, train_loss, train_acc, valid_loss, valid_acc))

            if epoch % 3 == 0:
                self.saver.save(self.session, config.save_path, global_step=step)

    def close(self):
        self.filewriter_valid.close()
        self.filewriter_train.close()
        self.session.close()



if __name__ == '__main__':
    config = Config()
    model = Model(config)
    model.train()
    model.close()

