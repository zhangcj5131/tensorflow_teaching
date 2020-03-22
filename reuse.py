import tensorflow as tf


with tf.variable_scope('test', reuse=None):
    x = tf.get_variable('x', [], tf.int32)
    print(x)