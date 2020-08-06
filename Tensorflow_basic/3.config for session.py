
import tensorflow as tf
import numpy as np

"""
介绍 tf.Session() 中参数config协议。
"""

def session_config():
    # 1、构建2个常量的矩阵
    a = tf.constant(value=[1,2,3,],
                    dtype=tf.float32, shape=[3, 5])
    b = tf.constant(value=[2,3,4],
                    dtype=tf.float32, shape=[5, 3])
    # 2、2个常量tensor分别加一个随机数
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[], dtype=tf.float32)

    # 3、矩阵相乘
    result = tf.matmul(v1, v2)
    print(v1, v2, result)
    # 二、构建会话
    gpu_options = tf.GPUOptions(
        allow_growth=True,             # 不预先分配整个gpu显存计算，而是从小到大，按需增加。
        per_process_gpu_memory_fraction=0.9  # value-(0, 1) 限制使用该gpu设备的显存使用的百分比。一般建议设置0.8--0.9左右。
    )
    config = tf.ConfigProto(
        allow_soft_placement=True,     # 是否允许tf动态的使用cpu和gpu。当我们的版本是gpu版本，那么tf会默认调用你的gpu:0来运算，如果你的gpu无法工作，那么tf就会报错。所以建议有gpu版本的同学，将这个参数设置为True。
        log_device_placement=True,      # bool值，是否打印设备位置的日志文件。
        gpu_options=gpu_options
    )
    with tf.Session(config=config) as sess:
        v1_, v2_, rezult_ = sess.run([v1, v2, result])
        print(tf.get_default_session())
        print(v1_, v2_, rezult_)


if __name__ == '__main__':
    session_config()

