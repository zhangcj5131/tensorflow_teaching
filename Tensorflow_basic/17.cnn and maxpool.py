import tensorflow as tf

def conv_test():
    with tf.Graph().as_default():
        x = tf.ones(shape = [64, 32, 32, 3])
        filter_w = tf.get_variable('w', shape = [3, 3, 3, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
        filter_b = tf.get_variable('b', shape = [10], dtype=tf.float32, initializer=tf.zeros_initializer())
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        conv = tf.nn.conv2d(input = x, filter=filter_w, strides=strides, padding = padding)
        conv = conv + filter_b
        conv = tf.nn.relu(conv)
        print(conv)

def pool_test():
    with tf.Graph().as_default():
        x = tf.ones(shape=[64, 32, 32, 3])
        pool = tf.nn.max_pool(value = x, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
        print(pool)
if __name__ == '__main__':
    pool_test()