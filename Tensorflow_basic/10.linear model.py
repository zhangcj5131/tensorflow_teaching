import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_test_data(n = 100):
    x = np.linspace(0, 5, n) + np.random.normal(size = n)
    y = x * 2 + 3 + np.random.normal(size = n)
    x.shape = -1, 1
    y.shape = -1, 1
    return x, y

def linear_model():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 1], 'x')
        y = tf.placeholder(tf.float32, [None, 1], 'y')

        w = tf.get_variable('w', [1,1], tf.float32, initializer=tf.random_normal_initializer)
        b = tf.get_variable('b', [1], tf.float32, initializer=tf.zeros_initializer)

        y_pred = tf.matmul(x, w) + b
        loss = tf.reduce_mean(tf.square(y - y_pred))
        tf.summary.scalar('loss', loss)

        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_train, y_train = train_test_data()
            feed_dict = {x: x_train, y: y_train}
            file_writer = tf.summary.FileWriter(logdir='./log/10', graph=sess.graph)
            step = 1
            for e in range(300):
                _,su, lo = sess.run([train_op, summary_op, loss], feed_dict)
                file_writer.add_summary(su, step)
                step += 1
                print('epoch=%d, loss=%f' % (e, lo))
            file_writer.close()
            y_predict = sess.run(y_pred, {x: x_train})
            plt.plot(x_train, y_train, 'r+')
            plt.plot(x_train, y_predict, 'g-')
            plt.show()



if __name__ == '__main__':
    linear_model()