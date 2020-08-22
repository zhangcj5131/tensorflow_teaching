import tensorflow as tf

"""
学习 卷积和池化 api
"""

def conv2d_answer():
    with tf.Graph().as_default():
        x = tf.ones(shape=[64, 32, 32, 3])  # [batch_size, height, width, channels=depth]

        # todo 创建卷积核变量
        filter_w = tf.get_variable(
            'w', initializer=tf.truncated_normal(shape=[7, 7, 3, 20], dtype=tf.float32)
        )
        filter_b = tf.get_variable(
            'b', initializer=tf.zeros(20)
        )
        strides = [1, 2, 2, 1]
        pad = 'SAME'
        """
        conv2d(input,   # 输入。注意：格式必须为：4-d tensor [batch_size, height, width, channels]
            filter,     # 卷积核。 格式为: [卷积核的高， 卷积核的宽， 卷积核的深度(输入图片的channels)， 卷积核的个数(输出图片的channels)]
            strides,    # 步幅。 [1, 高方向的步幅， 宽方向上的步幅， 1] 
            padding,    # 填充方式。 string类型，  'VALID' or 'SAME'
            use_cudnn_on_gpu=True, 
            data_format="NHWC",   # 对输入数据格式的要求。 N - batch_size, H- height, W - width  C-channels
            dilations=[1, 1, 1, 1],  # 膨胀因子。
            name=None):
        """
        conv_out = tf.nn.conv2d(
            input=x, filter=filter_w, strides=strides, padding=pad
        )
        conv_out = conv_out + filter_b
        conv_out = tf.nn.relu(conv_out)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(conv_out).shape)


def maxpool():
    with tf.Graph().as_default():
        x = tf.ones(shape=[64, 4, 4, 3])   # [N, H, W, C]

        max_out = tf.nn.max_pool(
            value=x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'
        )
        print(max_out)
        with tf.Session() as sess:
            print(sess.run(max_out))


if __name__ == '__main__':
    conv2d_answer()
    # maxpool()