import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

NUM = 200
HIDDEN_UNITS = 200
LINES = 2
def get_samples(num = NUM):
    start = -math.pi
    delta = (2*math.pi) / (num - 1)
    result = []
    for _ in range(num):
        result.append((start, math.sin(start), start ** 2))
        start += delta
    return result

#xs:[200]
#ys:[200, 2}
def plot(*para):#(xs, ys)
    for x, y in para:
        plt.plot(x, y)

def predict(xs, ys, lr = 0.01, epoches = 2000, hidden_units = HIDDEN_UNITS):
    x = tf.placeholder(tf.float32, [None], 'x')
    m = tf.layers.dense(tf.reshape(x, [-1, 1]), hidden_units, activation=tf.nn.relu, name = 'dense1')

    y_predict = tf.layers.dense(m, LINES, use_bias=False, name = 'dense2')#200,2


    y = tf.placeholder(tf.float32, [None, LINES], 'y')
    loss = tf.reduce_mean(tf.square(y - y_predict))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    loss = tf.sqrt(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(epoches):
            feed_dict = {
                x: xs,
                y: ys
            }
            _, lo = sess.run([train_op, loss], feed_dict)
            print('loss = %.8f' % lo)
        return sess.run(y_predict, {x: xs})

if __name__ == '__main__':
    samples = get_samples()
    t = np.transpose(samples, [1, 0])
    xs = t[0]
    ys = [t[i] for i in range(1, LINES + 1)]#2, 200
    ys = np.transpose(ys, [1, 0])#200, 2
    plot((xs, ys))

    y_predict = predict(xs, ys)
    plot((xs, y_predict))
    plt.show()