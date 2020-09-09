

import tensorflow as tf


def GroupNormaliztion(x, is_training, groups=32, decay=0.99, scope='gn'):
    """
    :param x:  x shape [N, H, W, C]
    :param is_training:  bool值， 批归一化参数
    :param groups:  分多少组
    :param scope:
    :return: [N, H, W, C]
    """
    with tf.variable_scope(scope):
        # 转置
        epsilon = 1e-5
        G = groups
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # [N, C, H, W]
        N, C, H, W = x.get_shape()
        G = min(G, C)  # 取groups 和 channels最小的值作为 分组。

        # 将x  reshape成 5-d tensor
        x = tf.reshape(x, shape=[-1, G, C//G, H, W])

        # 构建gamma 和 beta
        gamma = tf.get_variable('gamma', shape=[C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', shape=[C], initializer=tf.constant_initializer(0.0))

        gamma = tf.reshape(gamma, shape=[1, C, 1, 1])
        beta = tf.reshape(beta, shape=[1, C, 1, 1])

        # 构建全局统计量（用于推理的 移动平均数 和 移动平均方差）
        pop_mean = tf.Variable(initial_value=tf.zeros([N, G, 1, 1, 1]), trainable=False)
        pop_variance = tf.Variable(initial_value=tf.ones([N, G, 1, 1, 1]), trainable=False)

        # 计算当前批次的（按照分组）的  均值和 方差
        gn_mean, gn_variance = tf.nn.moments(x, axes=[2, 3, 4], keep_dims=True)
        # [N, G, 1, 1, 1]
        # 构建更新 全局统计量的操作符。
        train_mean = tf.assign(pop_mean, pop_mean*decay + (1-decay)*gn_mean)
        train_variance = tf.assign(pop_variance, pop_variance*decay + (1-decay)*gn_variance)

        def train():
            """
            构建训练的函数
            :return:
            """
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.identity(gn_mean), tf.identity(gn_variance)

        # 最终取决于参数is_training
        # 若is_training==True，代表训练，则更新全局统计量，返回 批次的统计量。
        # 反之，则返回 全局统计量。
        mean, variance = tf.cond(is_training, train, lambda: (pop_mean, pop_variance))

        # 计算分组归一化
        x = (x - mean) / tf.sqrt(variance + epsilon)  # [N, G, C//G, H, W]
        # reshape 并且进行缩放和位移
        output = tf.reshape(x, shape=[-1, C, H, W]) * gamma + beta  # [N, C, H, W]

        #  转置
        output = tf.transpose(output, perm=[0, 2, 3, 1])  # [N, H, W, C]
    return output


def test():
    data = tf.ones(shape=[2, 32, 32, 128], dtype=tf.float32)
    bn_train = tf.placeholder_with_default(True, shape=None, name='bn_training')
    gn_out = GroupNormaliztion(
        x=data, is_training=bn_train, groups=8
    )
    print(gn_out)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(gn_out))


if __name__ == '__main__':
    test()

