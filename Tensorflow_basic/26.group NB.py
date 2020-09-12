import tensorflow as tf


def group_BN(x, is_training, group = 8, eps = 1e-3, decay_rate = 0.99, name='group_bn'):
    with tf.variable_scope(name):
        x = tf.transpose(x, [0, 3, 1, 2])
        B, C, W, H = x.get_shape()
        G = min(C, group)
        x = tf.reshape(x, [B, G, C//G, W, H])

        gamma = tf.get_variable('gamma', dtype=tf.float32, shape = [1, C, 1, 1],
                                initializer=tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('beta', dtype=tf.float32, shape=[1, C, 1, 1],
                                initializer=tf.zeros_initializer(), trainable=True)
        mom_mean = tf.get_variable('mom_mean', dtype=tf.float32, shape=[B, G, 1, 1, 1],
                                initializer=tf.zeros_initializer(), trainable=False)
        mom_variance = tf.get_variable('mom_variance', dtype=tf.float32, shape=[B, G, 1, 1, 1],
                                initializer=tf.ones_initializer(), trainable=False)

        def train():
            #[B, G, 1, 1, 1]
            batch_mean, batch_variance = tf.nn.moments(x, axes=[2,3,4],keep_dims=True)
            assign_mean = tf.assign(
                mom_mean, decay_rate*mom_mean + (1-decay_rate)*batch_mean
            )
            assign_variance = tf.assign(
                mom_variance, decay_rate*mom_variance + (1-decay_rate)*batch_variance
            )
            with tf.control_dependencies(control_inputs=[assign_mean, assign_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = tf.cond(is_training, train, lambda: (mom_mean, mom_variance))
        output = (x - mean) / tf.sqrt(variance+eps)#[B, G, C//G, W, H]
        output = tf.reshape(output, [B, C, W, H])*gamma + beta
        output = tf.transpose(output, [0, 2, 3, 1])
    return output


def test():
    x = tf.ones(dtype=tf.float32, shape=[2, 32, 32, 128])
    is_training = tf.placeholder_with_default(True, [], 'is_training')
    group_bn = group_BN(x, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(group_bn.eval())#sess.run(group_bn)


if __name__ == '__main__':
    test()

