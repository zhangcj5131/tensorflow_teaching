import tensorflow as tf
import numpy as np


class Config:
    def __init__(self):
        self.lr = 0.001
        self.hidden_units = 200
        self.samples = 200
        self.epoches = 5000
        self.name = 'p03'
        self.save_path = './models/{name}/{name}'.format(name = self.name)




class Samples:
    def __init__(self, config: Config):
        start = 0
        end = 1
        delta = (end - start) / float(config.samples - 1)
        result = []
        while start <= 1:
            result.append((start**2, start))
            start += delta
        result = np.transpose(result, [1,0])
        self.x = result[0]
        self.y = result[1]

    def get_batch(self):
        return self.x, self.y


class Tensors:
    def __init__(self, config: Config):
        self.x = tf.placeholder(tf.float32, [None], 'x')#[batch_size]
        x = tf.expand_dims(self.x, axis = 1)#[batch_szie, 1]
        x = tf.layers.dense(x, config.hidden_units, activation=tf.nn.relu, name = 'dens1')#[batch_size,hidden_units]
        y_predict = tf.layers.dense(x, 1, name = 'dense2')#[batch_szie, 1]

        self.y_predict = tf.reshape(y_predict, [-1])#[batch_size]

        self.y = tf.placeholder(tf.float32, [None], 'y')#[batch_size]

        loss = tf.reduce_mean(tf.square(self.y - self.y_predict))

        self.lr = tf.placeholder(tf.float32, [], 'lr')

        opt = tf.train.AdamOptimizer(self.lr)
        self.train_opt = opt.minimize(loss)
        self.loss = tf.sqrt(loss)



class Sqr:
    def __init__(self):
        self.config = Config()

        self.tensors = Tensors(self.config)
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.session, self.config.save_path)
            print('successfully restored the model!')
        except:
            print('the model does not exist, we have to train a new one!')
            self.train()


    def train(self):
        self.samples = Samples(self.config)
        self.session.run(tf.global_variables_initializer())
        x, y = self.samples.get_batch()
        feed_dict = {
            self.tensors.x:x,
            self.tensors.y:y,
            self.tensors.lr: self.config.lr
        }
        for epoch in range(self.config.epoches):
            _, lo = self.session.run([self.tensors.train_opt, self.tensors.loss], feed_dict=feed_dict)
            print('epoch = %d, loss = %f' % (epoch, lo))
        self.saver.save(self.session, self.config.save_path)



    def predict(self,value):
        if type(value) not in (list, tuple):
            value = [value]

        feed_dict = {
            self.tensors.x: value
        }
        result = self.session.run(self.tensors.y_predict, feed_dict)
        return result

    def close(self):
        self.session.close()

if __name__ == '__main__':

    value = 3
    sqr = Sqr()
    # sqr.train()
    y_predict = sqr.predict(value/9)*3
    print('sqrt(%f)=%f' % (value, y_predict))
    sqr.close()