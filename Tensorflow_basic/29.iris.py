from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import os


class Config:
    def __init__(self):
        self.data_path = '../data/iris.data'
        self.lr = 0.01
        self.epoches = 300
        self.name = 29
        self.save_path = './models/{name}/{name}'.format(name = self.name)
        self.logdir = './log/{name}'.format(name=self.name)







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
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir,graph=graph)
            self.saver = tf.train.Saver(max_to_keep=2)
            dirname = os.path.dirname(config.save_path)
            ckpt = tf.train.get_checkpoint_state(dirname)
            if ckpt:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print('model was successfully restored')
            else:
                self.session.run(tf.global_variables_initializer())
                print('the model does not exist, we have to train a new one')

    def train(self):

        config = self.config
        samples = Samples(config)
        X_train, X_test, y_train, y_test = samples.get_data()
        step = 1
        for epoch in range(config.epoches+1):
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
            if epoch % 50 == 0:
                self.saver.save(self.session, config.save_path, global_step=epoch)
        test_dict = {
            self.tensors.x: X_test,
            self.tensors.y: y_test
        }
        acc = self.session.run(self.tensors.acc, feed_dict=test_dict)
        print('test acc = %f' % acc)

    def close(self):
        self.file_writer.close()
        self.session.close()



class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.ds = pd.read_csv(config.data_path, header=None)

    def data_explor(self):
        print(self.ds.head())
        print(self.ds[4].value_counts())
        print(self.ds[4].unique())

    def _preprocess_data(self):
        lable_map = {name: index for index, name in enumerate(self.ds[4].unique())}
        #print(lable_map)
        # print(self.ds.head())
        self.ds[4] = self.ds[4].map(lable_map)
        # print(self.ds.tail())
        #Series
        features = self.ds.iloc[:,0:4]
        labels = self.ds.iloc[:, 4]
        # print(labels.values)
        return features.values, labels.values

    def get_data(self):
        features, labels = self._preprocess_data()
        X_train, X_test, y_train, y_test = \
            train_test_split(features, labels, test_size=0.2, random_state=42)
        model = StandardScaler()
        X_train = model.fit_transform(X_train)
        X_test = model.transform(X_test)
        # print(X_train)
        return X_train, X_test, y_train, y_test


class Tensors:
    def __init__(self, config: Config):
        with tf.device('/gpu:0'):
            with tf.variable_scope('network'):
                self.x = tf.placeholder(tf.float32, [None, 4], 'x')
                self.y = tf.placeholder(tf.int32, [None], 'y')
                self.lr = tf.placeholder(tf.float32, [], 'lr')

                w = tf.get_variable('w', [4, 3], tf.float32,
                                    tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [3], tf.float32,
                                    tf.zeros_initializer())

                logits = tf.matmul(self.x, w) + b#None, 3
                y_pred = tf.math.argmax(logits, axis = 1, output_type=tf.int32)
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y, logits=logits
                ))
                tf.summary.scalar('loss', self.loss)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

                corr_num = tf.equal(self.y, y_pred)
                corr_num = tf.cast(corr_num, tf.float32)
                self.acc = tf.reduce_mean(corr_num)
                tf.summary.scalar('acc', self.acc)

                self.summary_op = tf.summary.merge_all()



if __name__ == '__main__':
    config = Config()
    # s = Samples(config)
    # s.get_train_test()
    # t = Tensors(config)
    model = Model(config)
    model.train()
    model.close()
