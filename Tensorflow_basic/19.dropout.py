import tensorflow as tf
import numpy as np


def my_dropout(x, keep_prob):
    x_shape = x.get_shape()
    print(x_shape)
    #防止为 0
    keep_prob = tf.maximum(keep_prob, 0.001)

    mask = tf.cast(tf.random_uniform(shape=x_shape, minval=0, maxval=1.) < keep_prob, tf.float32)/keep_prob
    return mask * x


def dropout_test():
    with tf.Graph().as_default():
        x = tf.constant(value = np.random.normal(size=[3,3]), dtype = tf.float32)
        w = tf.get_variable('w', dtype = tf.float32, shape=[3,2], initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', dtype=tf.float32, shape=[2], initializer=tf.zeros_initializer())
        output = tf.matmul(x, w) + b
        output_dropout = tf.nn.dropout(output,keep_prob=0.5)
        output_dropout1 = my_dropout(output, keep_prob=0.5)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_, output_dropout_, output_dropout1_ = sess.run([output, output_dropout, output_dropout1])
            print(output_)
            print('---')
            print(output_dropout_)
            print('---')
            print(output_dropout1_)
if __name__ == '__main__':
    dropout_test()