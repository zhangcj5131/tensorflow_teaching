import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


class Config:
    def __init__(self):
        self.data_path = '../data/iris.data'
        self.lr = 0.01
        self.epoches = 300
        self.name = 29
        self.save_path = './models/{name}/{name}'.format(name = self.name)
        self.logdir = './log/{name}'.format(name=self.name)


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        with tf.device('/gpu:0'):
            with tf.variable_scope('network'):
                self.x = tf.placeholder(tf.float32, [None, 4], 'x')
                self.y = tf.placeholder(tf.int32, [None], 'y')
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                w = tf.get_variable(name = 'w', shape = [4, 3], dtype = tf.float32,
                                    initializer = tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable(name='b', shape=[3], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                logits = tf.matmul(self.x, w) + b
                # prediction = tf.nn.softmax(logits)
                y_predict = tf.math.argmax(logits, axis = 1)
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y, logits=logits
                ))
                tf.summary.scalar('loss', self.loss)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                correct_num = tf.equal(tf.cast(y_predict, tf.int32), self.y)
                correct_num = tf.cast(correct_num, tf.float32)
                self.acc = tf.reduce_mean(correct_num)
                tf.summary.scalar('acc', self.acc)
                self.summary_op = tf.summary.merge_all()


class Model:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
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
            self.session = tf.Session(config=conf, graph=graph)
            self.tensors = Tensors(config)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir,
                                                     graph=graph)
            self.saver = tf.train.Saver(max_to_keep=2)
            try:
                dirname = os.path.dirname(config.save_path)
                ckpt = tf.train.get_checkpoint_state(dirname)
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print('model is successfully restored!')
            except:
                self.session.run(tf.global_variables_initializer())
                print('the model does not exist, we have to train a new one!')

    def train(self):
        config = self.config
        self.samples = Samples(config)
        X_train, X_test, y_train, y_test = self.samples.preprocess_data()
        step = 1
        for epoch in range(config.epoches):
            feed_dict = {self.tensors.x: X_train,
                         self.tensors.y: y_train,
                         self.tensors.lr: config.lr}
            _, lo, acc, su = self.session.run([self.tensors.train_op,
                              self.tensors.loss,
                              self.tensors.acc,
                              self.tensors.summary_op], feed_dict=feed_dict)
            self.file_writer.add_summary(su, global_step=step)
            step += 1
            print('epoch=%d, loss=%f, acc=%f' % (epoch, lo, acc))
            if epoch%50 == 0:
                self.saver.save(self.session, config.save_path, global_step=epoch)
                step+=1

        test_dict = {self.tensors.x: X_test,
                     self.tensors.y: y_test}
        test_acc = self.session.run(self.tensors.acc, feed_dict=test_dict)
        print('test acc = %f', test_acc)

    def close(self):
        self.file_writer.close()
        self.session.close()



class Samples:
    def __init__(self, config: Config):
        self._iris = pd.read_csv(config.data_path, header=None)

    def _read_data(self):
        label_map = {name: index for index, name in enumerate(self._iris[4].unique())}
        self._iris[4] = self._iris[4].map(label_map)
        features = self._iris.iloc[:, 0:4]
        labels = self._iris.iloc[:, 4]
        return features.values, labels.values

    def preprocess_data(self):
        features, labels = self._read_data()
        X_train, X_test, y_train, y_test = \
            train_test_split(features, labels, test_size=0.2, random_state=42)
        normal_func = StandardScaler()
        X_train = normal_func.fit_transform(X_train)
        X_test = normal_func.transform(X_test)
        return X_train, X_test, y_train, y_test




if __name__ == '__main__':
    config = Config()
    # s = Samples(config)
    # s.preprocess_data()
    #
    model = Model(config)
    model.train()
    model.close()