import numpy as np
import tensorflow as tf

#4*3
x = [[1,2,3],
     [2,3,4],
     [3,4,5],
     [4,5,6]]
#3*1
w = [[1],
     [2],
     [3]]
b = 2
#4*1
y= np.dot(x, w) + b

def f1():
    x = tf.constant(value = [[1,2,3],
     [2,3,4],
     [3,4,5],
     [4,5,6]], dtype = tf.float32, shape = [4,3], name = 'x')

    w = tf.constant(value = [[1],
     [2],
     [3]], dtype = tf.float32, name = 'w')

    b = tf.constant(value = 2, dtype = tf.float32, shape = [], name = 'b')

    y_pred = tf.matmul(x, w) + b
    with tf.Session() as sess:
        y_pred_=sess.run(y_pred)
        print(y_pred_)


def f2():
    x = tf.constant(value = [[1,2,3],
     [2,3,4],
     [3,4,5],
     [4,5,6]], dtype = tf.float32, shape = [4,3], name = 'x')


    w = tf.Variable(initial_value=[[1],
     [2],
     [3]], dtype = tf.float32, name = 'w')

    b1 = tf.constant(value = 2, dtype = tf.float32, shape = [], name = 'b1')
    b = tf.Variable(initial_value = b1, dtype=tf.float32,  name = 'b')

    y_pred = tf.matmul(x, w) + b
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_pred_=sess.run(y_pred)
        print(y_pred_)

def f3():
    x = tf.placeholder(dtype=tf.float32, shape = [None,3], name = 'x')
    c = tf.placeholder_with_default(input=2000., shape = [], name = 'c')

    w = tf.Variable(initial_value=[[1],
     [2],
     [3]], dtype = tf.float32, name = 'w')

    b1 = tf.constant(value = 2, dtype = tf.float32, shape = [], name = 'b1')
    b = tf.Variable(initial_value = b1, dtype=tf.float32,  name = 'b')

    y_pred = tf.matmul(x, w) + b + c
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_x = [[1,2,3],
     [2,3,4],
     [3,4,5],
     [4,5,6]]
        y_pred_=sess.run(y_pred, feed_dict={x: feed_x, c:0})
        print(y_pred_)

        feed_x = [[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]]
        y_pred_ = sess.run(y_pred, feed_dict={x: feed_x})
        print(y_pred_)

#tensorboard --logdir=/Users/cjz/PycharmProjects/tensorflow_teaching/Tensorflow_basic/models/4
def f4():
    x = tf.placeholder(dtype=tf.float32, shape = [None,3], name = 'x')
    c = tf.placeholder_with_default(input=2000., shape = [], name = 'c')

    w = tf.Variable(initial_value=[[1],
     [2],
     [3]], dtype = tf.float32, name = 'w')

    b1 = tf.constant(value = 2, dtype = tf.float32, shape = [], name = 'b1')
    b = tf.Variable(initial_value = b1, dtype=tf.float32,  name = 'b')

    y_pred = tf.matmul(x, w) + b + c
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        file_writer = tf.summary.FileWriter(logdir='./models/9', graph=sess.graph)
        feed_x = [[1,2,3],
     [2,3,4],
     [3,4,5],
     [4,5,6]]
        y_pred_=sess.run(y_pred, feed_dict={x: feed_x, c:0})
        print(y_pred_)

        feed_x = [[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]]
        y_pred_ = sess.run(y_pred, feed_dict={x: feed_x})
        print(y_pred_)
        file_writer.close()






if __name__ == '__main__':
    f4()


