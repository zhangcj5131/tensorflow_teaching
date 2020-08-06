
import tensorflow as tf

"""
学习tf.device()添加运行设备号
注意：
  如果不使用tf.device()来给定具体的运行设备，那么tf会根据你的tf版本来选择默认的设备进行运算。
  1、如果时tf cpu版本，那么运行再cpu上。
  2、如果是tensorflow-gpu版本，那么运算操作一定会在第一块gpu上运行，在所有gpu上面分配内存。
  如果你通过tf.device()明确指定运行设备，那么运算操作一定会在你指定的设备上运行，如果没有该设备，就会报错。
  这时，可以通过 Session中的参数，allow_soft_placement=True来设置。
"""


def show_device():
    with tf.Graph().as_default():
        with tf.device('/CPU:0'):
            x = tf.placeholder(tf.float32, [], 'x')
            n = tf.Variable(initial_value=0., dtype=tf.float32, name='n')

        with tf.device('/GPU:1'):
            n_assign = tf.assign(n, n + x)

        with tf.device('/GPU:0'):
            with tf.control_dependencies([n_assign]):
                y = n * 1
        conf = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement = True
        )
        with tf.Session(config=conf) as sess:
            sess.run(tf.global_variables_initializer())
            for i in [1,2,3,4]:
                y_ = sess.run(y, feed_dict={x: i})
                print(y_)



if __name__ == '__main__':
    show_device()