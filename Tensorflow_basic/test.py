
import tensorflow as tf

"""
学习 relu激活函数 和 dropout使用
"""

tf.set_random_seed(43)

def activation_func():
    with tf.Graph().as_default():
        hidden_layer_weights = tf.truncated_normal(shape=[4, 3], stddev=1.0)
        output_weights = tf.truncated_normal(shape=[3, 2], stddev=1.0)

        weights = [
            tf.get_variable('w1', initializer=hidden_layer_weights),
            tf.get_variable('w2', initializer=output_weights)
        ]
        biases = [
            tf.get_variable('b1', initializer=tf.zeros(3)),
            tf.get_variable('b2', initializer=tf.zeros(2))
        ]

        # 构建输入
        input_x = tf.constant(value=[[1.0, 2.2, 3.4, 3.5],
                                     [-1.2, -2.3, -8.9, -9.0],
                                     [1.0, 2.2, 3.4, 3.5]], dtype=tf.float32)

        hidden_input = tf.matmul(input_x, weights[0]) + biases[0]
        hidden_output = tf.nn.relu(hidden_input)
        hidden_output1 = tf.nn.relu6(hidden_input)
        hidden_output2 = tf.nn.leaky_relu(hidden_input, alpha=0.02)
        # 自己实现leaky RELU
        alpha = 0.2
        hidden_output3 = tf.maximum(alpha*hidden_input, hidden_input)

        final_inputs = tf.matmul(hidden_output, weights[1]) + biases[1]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            hidden_input_, hidden_output_, hidden_output1_, hidden_output2_, hidden_output3_ = sess.run(
                [hidden_input, hidden_output, hidden_output1, hidden_output2, hidden_output3])
            print(hidden_input_,
                  '\n',
                  hidden_output_, '\n',
                  hidden_output1_, '\n',
                  hidden_output2_, '\n',
                  hidden_output3_)

def drop_out(input_tensor, keep_prob=0.6):
    # x_shape = input_tensor.get_shape()  # get_shape() 给定的tensor是确定的 （没有使用None）
    x_shape = tf.shape(input_tensor)
    epsilon = 0.0000001
    keep_prob = tf.maximum(keep_prob, epsilon)
    keep_prob = tf.cast(tf.random_uniform(shape=x_shape, maxval=1.0) < keep_prob, tf.float32) / keep_prob
    drop_output = input_tensor * keep_prob
    return drop_output


def show_dropout():
    with tf.Graph().as_default():
        hidden_layer_weights = tf.truncated_normal(shape=[4, 3], stddev=1.0)
        output_weights = tf.truncated_normal(shape=[3, 2], stddev=1.0)

        weights = [
            tf.get_variable('w1', initializer=hidden_layer_weights),
            tf.get_variable('w2', initializer=output_weights)
        ]
        biases = [
            tf.get_variable('b1', initializer=tf.zeros(3)),
            tf.get_variable('b2', initializer=tf.zeros(2))
        ]

        # 构建输入
        input_x = tf.constant(value=[[1.0, 2.2, 3.4, 3.5],
                                     [-1.2, -2.3, -8.9, -9.0],
                                     [1.0, 2.2, 3.4, 3.5]], dtype=tf.float32)

        # 自己实现leaky RELU
        hidden_input = tf.matmul(input_x, weights[0]) + biases[0]
        alpha = 0.2
        hidden_output = tf.maximum(alpha*hidden_input, hidden_input)
        # 使用dropout
        hidden_output_drop = tf.nn.dropout(hidden_output, keep_prob=0.5)
        hidden_output_drop1 = drop_out(hidden_output, keep_prob=0.5)

        final_inputs = tf.matmul(hidden_output_drop, weights[1]) + biases[1]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            hidden_output_, hidden_output_drop_, hidden_output_drop1_ = sess.run(
                [hidden_output, hidden_output_drop, hidden_output_drop1])
            print(hidden_output_)
            print('-------')
            print(hidden_output_drop_)
            print('-------')
            print(hidden_output_drop1_)


if __name__ == '__main__':
    # activation_func()
    show_dropout()