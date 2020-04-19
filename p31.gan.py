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

        self.name = 'p31'
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
            grad1 = self.merge_grads(lambda ts: ts.grad1)
            grad2 = self.merge_grads(lambda ts: ts.grad2)
            grad3 = self.merge_grads(lambda ts: ts.grad3)

            self.train_op1 = opt.apply_gradients(grad1)
            self.train_op2 = opt.apply_gradients(grad2)
            self.train_op3 = opt.apply_gradients(grad3)

            self.loss1 = tf.reduce_mean([ts.loss1 for ts in self.sub_tensors])
            self.loss2 = tf.reduce_mean([ts.loss2 for ts in self.sub_tensors])
            self.loss3 = tf.reduce_mean([ts.loss3 for ts in self.sub_tensors])

            tf.summary.scalar('loss1', self.loss1)
            tf.summary.scalar('loss2', self.loss2)
            tf.summary.scalar('loss3', self.loss3)

            self.summary_op = tf.summary.merge_all()


    def merge_grads(self, func):
        indices_grads = {}
        grads = {}
        for ts in self.sub_tensors:
            for g, v in func(ts):
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
            with tf.variable_scope('discriminator'):
                self.x = tf.placeholder(tf.float32, [None, config.size, config.size], 'x')
                x = self.x / 255
                x = tf.reshape(x, [-1, config.size, config.size, 1])
                p1 = self.discriminator(x)

            with tf.variable_scope('geneator'):
                self.z = tf.placeholder(tf.float32, [None, config.z_size], 'z')
                fake = self.geneator(self.z)
                self.fake = tf.reshape(fake*255, [-1, config.size, config.size])

            with tf.variable_scope('discriminator', reuse=True):
                p2 = self.discriminator(fake)
            #-1*log(p) - 0*log(1-p)
            self.loss1 = - tf.reduce_mean(tf.log(p1))
            self.loss2 = - tf.reduce_mean(tf.log(1-p2))
            self.loss3 = - tf.reduce_mean(tf.log(p2))

            dis_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            gen_vars = [var for var in tf.trainable_variables() if 'geneator' in var.name]

            self.grad1 = opt.compute_gradients(self.loss1, dis_vars)
            self.grad2 = opt.compute_gradients(self.loss2, dis_vars)
            self.grad3 = opt.compute_gradients(self.loss3, gen_vars)

    def discriminator(self, x):
        config = self.config
        filters = config.filters
        for i in range(config.convs):
            filters *= 2
            x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name = 'conv_%d' % i)
            x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.flatten(x)#4,4,16
        x = tf.layers.dense(x, 1000, activation = tf.nn.relu, name = 'dense1')
        x = tf.layers.dense(x, 1, name='dense2')
        p = tf.nn.sigmoid(x)
        return p

    def geneator(self, z):
        config = self.config
        filters = config.filters
        filters = filters * int(2**config.convs)
        size = config.size // (2 ** config.convs)
        z = tf.layers.dense(z, size*size*filters, name = 'dense')
        z = tf.reshape(z, [-1, size, size, filters])
        for i in range(config.convs):
            filters //=2
            z = tf.layers.conv2d_transpose(z, filters, 3, 2, 'same', activation=tf.nn.relu, name = 'deconv_%d' % i)
        z = tf.layers.conv2d_transpose(z, 1, 3, 1, 'same', name = 'deconv')
        return z




            





class Samples:
    def __init__(self, config: Config):
        self.config = config
        self.index = 0
        self.data = []
        for _ in range(self.num()):
            self.data.append(self.get_sample())

    def num(self):
        return 100

    def next_batch(self, batch_size):
        next = self.index + batch_size
        if next < self.num():
            result = self.data[self.index: next]
        else:
            result = self.data[self.index:]
            next -= self.num()
            result += self.data[:next]
        self.index = next
        return result, np.random.normal(size=[batch_size, config.z_size])

    def get_sample(self):
        config = self.config
        img = np.zeros([config.size, config.size])
        points = np.random.randint(0, config.size, [2, 2])
        cv2.rectangle(img, tuple(points[0]), tuple(points[1]), 255, 1)
        return img






class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        self.samples = Samples(self.config)
        with graph.as_default():
            self.tensors = Tensors(config)
            self.file_writer = tf.summary.FileWriter(logdir=config.logdir, graph=graph)
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
        config = self.config
        batches = self.samples.num() // (config.gpus * config.batch_size)
        for epoch in range(config.epoches):
            for batch in range(batches):
                feed_dict = {
                    self.tensors.lr:config.lr
                }
                for gpu_index in range(config.gpus):
                    x, z = self.samples.next_batch(config.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].x] = x
                    feed_dict[self.tensors.sub_tensors[gpu_index].z] = z
                _, loss1 = self.session.run([self.tensors.train_op1, self.tensors.loss1], feed_dict)
                _, loss2 = self.session.run([self.tensors.train_op2, self.tensors.loss2], feed_dict)
                _, loss3 = self.session.run([self.tensors.train_op3, self.tensors.loss3], feed_dict)
                su = self.session.run(self.tensors.summary_op, feed_dict)
                print('%d/%d, loss1 = %f, loss2 = %f, loss3 = %f' % (epoch, batch, loss1, loss2, loss3))
                self.file_writer.add_summary(su, epoch*batches+batch)
            self.saver.save(self.session, config.save_path)






    def predict(self, batch_size):
        _, z = self.samples.next_batch(batch_size)
        feed_dict = {
            self.tensors.lr: self.config.lr
        }
        feed_dict[self.tensors.sub_tensors[0].z] = z
        imgs = self.session.run(self.tensors.sub_tensors[0].fake, feed_dict)  # [batch_size, size, size]
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


    Tensors(config)

    app = App(config)


    app.train()
    app.predict(1)

    app.close()
    print('Finished!')
