import tensorflow as tf
import numpy as np
import os
import argparse
import cv2


class Config:
    def __init__(self):
        self.batch_size = 20
        self.size = 32
        self.z_size = 2
        self.convs = 3
        self.filters = 2

        self.gpus = self.get_gpus()

        self.name = 'p32'
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
            grad1 = self.merge_grads(lambda ts: ts.grad1)
            grad2 = self.merge_grads(lambda ts: ts.grad2)
            grad3 = self.merge_grads(lambda ts: ts.grad3)

            self.train_op1 = opt.apply_gradients(grad1)
            self.train_op2 = opt.apply_gradients(grad2)
            self.train_op3 = opt.apply_gradients(grad3)

            self.loss1 = tf.reduce_mean([ts.loss1 for ts in self.sub_ts])
            self.loss2 = tf.reduce_mean([ts.loss2 for ts in self.sub_ts])
            self.loss3 = tf.reduce_mean([ts.loss3 for ts in self.sub_ts])

            tf.summary.scalar('loss1', self.loss1)
            tf.summary.scalar('loss2', self.loss2)
            tf.summary.scalar('loss3', self.loss3)
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

    def merge_grads(self, func):
        indexed_grads = {}
        grads = {}
        for ts in self.sub_ts:
            for g, v in func(ts):
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
            with tf.variable_scope('disc'):
                self.x = tf.placeholder(tf.float32, [None, config.size, config.size], 'x')
                x = tf.reshape(self.x / 255, [-1, config.size, config.size, 1])
                p1 = self.get_disc(x)

            with tf.variable_scope('gene'):
                self.z = tf.placeholder(tf.float32, [None, config.z_size], 'z')
                fake = self.get_gene(self.z)
                self.fake = tf.reshape(fake * 255, [-1, config.size, config.size])

            with tf.variable_scope('disc', reuse=True):
                p2 = self.get_disc(fake)

            #-1*log(p1)-0*log(1-p1)
            self.loss1 = -tf.reduce_mean(tf.log(p1))
            #-0*log(p2) -1*log(1-p2)
            self.loss2 = -tf.reduce_mean(tf.log(1 - p2))
            #-1*log(p2) - 0*log(1-p2)
            self.loss3 = -tf.reduce_mean(tf.log(p2))

            disc_vars = [var  for var in tf.trainable_variables() if 'disc' in var.name]
            gene_vars = [var  for var in tf.trainable_variables() if 'gene' in var.name]

            self.grad1 = opt.compute_gradients(self.loss1, disc_vars)
            self.grad2 = opt.compute_gradients(self.loss2, disc_vars)
            self.grad3 = opt.compute_gradients(self.loss3, gene_vars)

    def get_disc(self, img):
        # img: [size, size, 1]
        filters = self.config.filters
        size = self.config.size
        for i in range(self.config.convs):
            filters *= 2
            img = tf.layers.conv2d(img, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv2d_%d' % i)
            img = tf.layers.max_pooling2d(img, 2, 2)
            size //= 2

        img = tf.layers.flatten(img)  # [-1, 4, 4, 16]
        img = tf.layers.dense(img, 1, name='dense')  # [-1, 1]
        p = tf.nn.sigmoid(img)
        return p  # [-1, 1]

    def get_gene(self, z):
        #  z: [-1, z_size]
        cfg = self.config
        size = cfg.size // (2 ** cfg.convs)  # 4
        filters = cfg.filters * int(2**cfg.convs)  # 16
        z = tf.layers.dense(z, size * size * filters, name='dense1')
        z = tf.reshape(z, [-1, size, size, filters])

        for i in range(cfg.convs):
            filters //= 2
            size *= 2
            z = tf.layers.conv2d_transpose(z, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d' % i)

        # z: [32, 32, 2]
        z = tf.layers.conv2d_transpose(z, 1, 3, 1, 'same', name='deconv2')  # [32, 32, 1]
        return z



class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0

        self.data = []
        for _ in range(self.num()):
            self.data.append(self.get_sample())

    def num(self):
        """
        The number of the samples in total.
        :return:
        """
        return 100

    def get_sample(self):
        img = np.zeros([self.config.size, self.config.size], np.int32)
        points = np.random.randint(0, self.config.size, [2, 2])
        cv2.rectangle(img, tuple(points[0]), tuple(points[1]), 255, 1)
        return img

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < len(self.data):
            result = self.data[self.index: next]
        else:
            result = self.data[self.index:]
            next -= len(self.data)
            result += self.data[:next]
        self.index = next
        return result, np.random.normal(size=[batch_size, self.config.z_size])


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        self.samples = Samples(config)
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


        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)

        for epoch in range(cfg.epoches):
            batches = self.samples.num() // (cfg.gpus * cfg.batch_size)
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr: cfg.lr
                }

                for gpu_index in range(cfg.gpus):
                    x, z = self.samples.next_batch(cfg.batch_size)
                    feed_dict[self.tensors.sub_ts[gpu_index].x] = x
                    feed_dict[self.tensors.sub_ts[gpu_index].z] = z

                _, loss1 = self.session.run([self.tensors.train_op1, self.tensors.loss1], feed_dict)
                _, loss2 = self.session.run([self.tensors.train_op2, self.tensors.loss2], feed_dict)
                _, loss3, su = self.session.run(
                    [self.tensors.train_op3, self.tensors.loss3, self.tensors.summary_op], feed_dict)
                writer.add_summary(su, epoch * batches + batch)
                print('%d/%d: loss1=%.8f, loss2=%.8f, loss3=%.8f' % (batch, epoch, loss1, loss2, loss3), flush=True)
            self.saver.save(self.session, cfg.save_path)
            print('Save the mode into ', cfg.save_path, flush=True)

    def predict(self, batch_size):
        _, z = self.samples.next_batch(batch_size)
        feed_dict = {
            self.tensors.lr: self.config.lr
        }
        feed_dict[self.tensors.sub_ts[0].z] = z
        imgs = self.session.run(self.tensors.sub_ts[0].fake, feed_dict)  # [batch_size, size, size]
        imgs = np.reshape(imgs, [-1, self.config.size])
        imgs = np.uint8(imgs)
        cv2.imshow('imgs', imgs)
        cv2.waitKey()

    def close(self):
        self.session.close()


if __name__ == '__main__':
    config = Config()
    config.from_cmd_line()

    # s = Samples(config)
    # img = s.get_sample()
    # cv2.imshow('My pic', np.uint8(img))
    # cv2.waitKey()

    # config.gpus=2
    # Tensors(config)

    app = App(config)

    app.train()
    app.predict(1)
    app.close()
    print('Finished!')
