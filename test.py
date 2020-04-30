import tensorflow as tf
import matplotlib.pyplot as pt
import math
import numpy as np


LINES = 3
SAMPLES = 200
HIDDEN_UNITS= 200


def get_samples(num):
    start = -math.pi
    delta = (math.pi * 2) / (num - 1)
    result = []
    for _ in range(num):
        result.append((start, math.sin(start), math.cos(start), start**2/10))
        start += delta
    return result


def prepare_show(*xy_pairs_list):
    for xs, ys in xy_pairs_list:
        pt.plot(xs, ys)


def show():
    pt.show()


def predict(xs, ys, hidden_units=HIDDEN_UNITS, lr = 0.001, epoches = 2000):
    x = tf.placeholder(tf.float32, [None], 'x')
    m = tf.layers.dense(tf.reshape(x, [-1, 1]), hidden_units, activation=tf.nn.relu, name='dense')
    y_predict = tf.layers.dense(m, LINES, use_bias=False, name='dense2')
    y = tf.placeholder(tf.float32, [None, LINES], 'y')

    loss = tf.reduce_mean(tf.square((y - y_predict)))
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.minimize(loss)
    loss = tf.sqrt(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(epoches):
            feed_dict = {
                x: xs,
                y: ys
            }
            _, los = session.run([train_op, loss], feed_dict)
            print('loss: %.8f' % los, flush=True)

        return session.run(y_predict, {x: xs})


if __name__ == '__main__':
    samples = get_samples(SAMPLES)
    samples = np.array(samples)[:, 0:LINES+1]
    t = np.transpose(samples, [1, 0])
    xs = t[0]
    ys = [t[i] for i in range(1, LINES+1)]
    print(ys)
    ys = np.transpose(ys, [1, 0])

    ys_predict = predict(xs, ys)
    ys_predict = np.transpose(ys_predict, [1, 0])
    prepare_show((xs, ys))
    for ys_p in ys_predict:
        prepare_show((xs, ys_p))
    show()