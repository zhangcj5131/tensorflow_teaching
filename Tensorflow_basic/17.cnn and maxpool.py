import tensorflow as tf

def conv_test():
    with tf.Graph().as_default():
        x = tf.ones(shape = [64, 32, 32, 3])
        #选择 10 个卷积核
        filter_w = tf.get_variable('w', shape = [3, 3, 3, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
        filter_b = tf.get_variable('b', shape = [10], dtype=tf.float32, initializer=tf.zeros_initializer())
        #1.batch 维度,固定为 1.   2,3.图片宽度和高度上的步长,通常设置为相同的值. 4.通道上的步长,固定为 1
        strides = [1, 1, 1, 1]
        padding = 'SAME'
        conv = tf.nn.conv2d(input = x, filter=filter_w, strides=strides, padding = padding)
        conv = conv + filter_b
        conv = tf.nn.relu(conv)
        print(conv)

#池化操作只修改长宽,不修改通道数
def pool_test():
    with tf.Graph().as_default():
        x = tf.ones(shape=[64, 32, 32, 3])
        pool = tf.nn.max_pool(value = x, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
        print(pool)
if __name__ == '__main__':
    pool_test()