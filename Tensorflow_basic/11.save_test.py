import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
def get_data(n=100):
    x = np.linspace(0,10,n)
    y = 2 * x + 3 + np.random.normal(size=n)
    x.shape = -1, 1
    y.shape = -1, 1
    return x, y

class Config:
    def __init__(self):
        self.name = 11
        self.lr = 0.01
        self.epoch = 301
        self.save_path = './models/{name}/{name}'.format(name=self.name)
        self.log_dir = './log/{name}'.format(name = self.name)


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        with tf.device('/gpu:0'):
            with tf.variable_scope('network'):
                self.x = tf.placeholder(tf.float32, [None, 1], 'x')
                self.y = tf.placeholder(tf.float32, [None, 1], 'y')
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                w = tf.get_variable('w', [1,1], tf.float32,tf.random_normal_initializer)
                b = tf.get_variable('b', [1], initializer=tf.zeros_initializer, dtype=tf.float32)
                self.y_pred = tf.matmul(self.x, w) + b

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.y - self.y_pred))
                tf.summary.scalar('loss', self.loss)

            with tf.variable_scope('optimizer'):
                self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
                self.summary_op = tf.summary.merge_all()


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
            # conf = tf.ConfigProto()
            # conf.allow_soft_placement = True

            self.session = tf.Session(config=conf, graph=graph)
            self.tensors = Tensors(config)
            #这个 saver 对象必须定义在 tensor 后面,否则会报错,因为模型都不存在,saver 不知道保存什么
            self.saver = tf.train.Saver(max_to_keep=2)
            self.file_writer = tf.summary.FileWriter(logdir=config.log_dir, graph=graph)




            try:
                # self.saver.restore(self.session, config.save_path)
                #获得路径
                dirname = os.path.dirname(config.save_path)
                #如果模型不存在,ckpt 会是 null,触发空指针异常
                ckpt = tf.train.get_checkpoint_state(dirname)
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print('model was successfully restored!')
            except:
                print('the model does not exist, we have to train a new one!')
                self.train()

    def train(self):
        self.session.run(tf.global_variables_initializer())
        x, y = get_data()
        feed_dict = {self.tensors.x: x,
                     self.tensors.y: y,
                     self.tensors.lr: self.config.lr}
        step = 1
        for e in range(self.config.epoch):
            _, su, lo = self.session.run([self.tensors.train_op,
                              self.tensors.summary_op,
                              self.tensors.loss], feed_dict=feed_dict)
            self.file_writer.add_summary(su, step)
            step += 1
            print('epoch=%d, loss=%f' % (e, lo))
            if e % 100 == 0:
                self.saver.save(self.session, self.config.save_path, global_step=e)
        # self.saver.save(self.session, self.config.save_path)


    def predict(self):
        x, y = get_data()
        feed_dict = {self.tensors.x: x}
        y_predict = self.session.run(self.tensors.y_pred, feed_dict=feed_dict)
        plt.plot(x, y, 'r+')
        plt.plot(x, y_predict, 'g-')
        plt.show()


    def close(self):
        self.file_writer.close()
        self.session.close()




if __name__ == '__main__':
    config = Config()
    model = Model(config)
    model.predict()
    model.close()