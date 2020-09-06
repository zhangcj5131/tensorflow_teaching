


"""
本小节目的：分别运用下面2种API来 完成BN的训练。并比较做了BN和 未做BN的网络的准确率的区别。

1. [Batch Normalization with `tf.layers.batch_normalization`](#example_1)
2. [Batch Normalization with `tf.nn.batch_normalization`](#example_2)
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True, reshape=False)


# todo-未加BN的代码，直接copy演示。
def fully_connected(prev_layer, num_units):
    """
    Create a fully connectd layer with the given layer as input and the given number of neurons.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param num_units: int
        The size of the layer. That is, the number of units, nodes, or neurons.
    :returns Tensor
        A new fully connected layer
    """
    layer = tf.layers.dense(prev_layer, num_units, activation=tf.nn.relu)
    return layer


def conv_layer(prev_layer, layer_depth):
    """
    Create a convolutional layer with the given layer as input.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param layer_depth: int
        We'll set the strides and number of feature maps based on the layer's depth in the network.
        This is *not* a good way to make a CNN, but it helps us create this example with very little code.
    :returns Tensor
        A new convolutional layer
    """
    strides = 2 if layer_depth % 3 == 0 else 1
    conv_layer = tf.layers.conv2d(prev_layer, layer_depth * 4, 3, strides, 'same', activation=tf.nn.relu)
    return conv_layer


def train(num_batches, batch_size, learning_rate):
    # Build placeholders for the input samples and labels
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    # Feed the inputs into a series of 20 convolutional layers
    layer = inputs
    for layer_i in range(1, 20):
        layer = conv_layer(layer, layer_i)

    # Flatten the output from the convolutional layers
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

    # Add one fully connected layer
    layer = fully_connected(layer, 100)

    # Create the output layer with 1 node for each
    logits = tf.layers.dense(layer, 10)

    # Define
    model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels
    ))

    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train and test the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_i in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # train this batch
            sess.run(train_opt, {inputs: batch_xs,
                                 labels: batch_ys})

            # Periodically check the validation or training loss and accuracy
            if batch_i % 100 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: mnist.validation.images,
                                                              labels: mnist.validation.labels})
                print(
                    'Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            elif batch_i % 25 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

        # At the end, score the final accuracy for both the validation and test sets
        acc = sess.run(accuracy, {inputs: mnist.validation.images,
                                  labels: mnist.validation.labels})
        print('Final validation accuracy: {:>3.5f}'.format(acc))
        acc = sess.run(accuracy, {inputs: mnist.test.images,
                                  labels: mnist.test.labels})
        print('Final test accuracy: {:>3.5f}'.format(acc))

        # Score the first 100 test images individually, just to make sure batch normalization really worked
        correct = 0
        for i in range(100):
            correct += sess.run(accuracy, feed_dict={inputs: [mnist.test.images[i]],
                                                     labels: [mnist.test.labels[i]]})

        print("Accuracy on 100 samples:", correct / 100)

# num_batches = 800
# batch_size = 64
# learning_rate = 0.002
#
# tf.reset_default_graph()
# with tf.Graph().as_default():
#     train(num_batches, batch_size, learning_rate)

# todo - 下面开始使用BN(使用`tf.layers.batch_normalization`)进行训练。

def fully_connected(prev_layer, num_units, is_training):
    """
    Create a fully connectd layer with the given layer as input and the given number of neurons.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param num_units: int
        The size of the layer. That is, the number of units, nodes, or neurons.
    :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
    :returns Tensor
        A new fully connected layer
    """
    layer = tf.layers.dense(
        prev_layer, num_units, use_bias=False, activation=None
    )
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def conv_layer(prev_layer, layer_depth, is_training):
    """
    Create a convolutional layer with the given layer as input.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param layer_depth: int
        We'll set the strides and number of feature maps based on the layer's depth in the network.
        This is *not* a good way to make a CNN, but it helps us create this example with very little code.
    :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
    :returns Tensor
        A new convolutional layer
    """
    strides = 2 if layer_depth % 3 == 0 else 1
    # fixme 因为Bnorm中有 位移因子beta（它就相当于偏置项的功能）， 所以卷积中不用使用bias。
    conv_layer = tf.layers.conv2d(
        prev_layer, layer_depth * 4, 3, strides, 'same', use_bias=False, activation=None
    )
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.relu(conv_layer)

    return conv_layer


def train_BN(num_batches, batch_size, learning_rate):
    # Build placeholders for the input samples and labels
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    # Add placeholder to indicate whether or not we're training the model
    is_training = tf.placeholder(tf.bool)

    # Feed the inputs into a series of 20 convolutional layers
    layer = inputs
    for layer_i in range(1, 20):
        layer = conv_layer(layer, layer_i, is_training)

    # Flatten the output from the convolutional layers
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

    # Add one fully connected layer
    layer = fully_connected(layer, 100, is_training)

    # Create the output layer with 1 node for each
    logits = tf.layers.dense(layer, 10)

    # Define loss and training operations
    model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels
    ))

    # Tell TensorFlow to update the population statistics while training
    # fixme 有BN都必须如此使用，控制依赖项中含义是 先更新全局统计量（均值和方差的移动平均数）
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    # Create operations to test accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train and test the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_i in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # 执行梯度下降
            sess.run(train_opt, {inputs: batch_xs, labels: batch_ys, is_training: True})

            # Periodically check the validation or training loss and accuracy
            if batch_i % 100 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: mnist.validation.images,
                                                              labels: mnist.validation.labels,
                                                              is_training: False})
                print(
                    'Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            elif batch_i % 25 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys, is_training: False})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

        # At the end, score the final accuracy for both the validation and test sets
        acc = sess.run(accuracy, {inputs: mnist.validation.images,
                                  labels: mnist.validation.labels,
                                  is_training: False})
        print('Final validation accuracy: {:>3.5f}'.format(acc))
        acc = sess.run(accuracy, {inputs: mnist.test.images,
                                  labels: mnist.test.labels,
                                  is_training: False})
        print('Final test accuracy: {:>3.5f}'.format(acc))

        # Score the first 100 test images individually, just to make sure batch normalization really worked
        correct = 0
        for i in range(100):
            correct += sess.run(accuracy, feed_dict={inputs: [mnist.test.images[i]],
                                                     labels: [mnist.test.labels[i]],
                                                     is_training: False})

        print("Accuracy on 100 samples:", correct / 100)

#
# num_batches = 800
# batch_size = 64
# learning_rate = 0.002
#
# tf.reset_default_graph()
# with tf.Graph().as_default():
#     train_BN(num_batches, batch_size, learning_rate)


# todo - 下面开始自己实现BN，并进行训练。
def fully_connected(prev_layer, num_units, is_training):
    """
    Create a fully connectd layer with the given layer as input and the given number of neurons.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param num_units: int
        The size of the layer. That is, the number of units, nodes, or neurons.
    :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
    :returns Tensor
        A new fully connected layer
    """

    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)

    gamma = tf.Variable(initial_value=tf.ones([num_units]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([num_units]), trainable=True)

    # 仅仅作为计算存储用(用于预测的)，不参与模型训练。
    pop_mean = tf.Variable(initial_value=tf.zeros([num_units]), trainable=False)
    pop_variance = tf.Variable(initial_value=tf.ones([num_units]), trainable=False)

    epsilon = 1e-3  # 防止除以0的。

    def batch_norm_training():
        """
        用于训练阶段的。
        :return:
        """
        # 1、tf.nn.moments 计算`x`的均值和方差.
        batch_mean, batch_variance = tf.nn.moments(layer, axes=[0])
        # 2、构建更新 全局统计量的assign赋值操作符
        decay = 0.99
        train_mean = tf.assign(
            ref=pop_mean, value=pop_mean * decay + (1-decay) * batch_mean
        )
        train_variance = tf.assign(
            pop_variance, value=pop_variance * decay + (1-decay) * batch_variance
        )
        # 3、构建控制依赖项，并最终执行当前批次的批归一化操作。
        with tf.control_dependencies(control_inputs=[train_mean, train_variance]):
            # 执行当前批量数据的归一化
            normalized_linear_output = (layer - batch_mean) / tf.sqrt(batch_variance + epsilon)
            return normalized_linear_output * gamma + beta

    def batch_norm_inference():
        """
        用于推理（验证 和 测试的时候）
        :return:
        """
        normalized_linear_output = (layer - pop_mean) / tf.sqrt(pop_variance + epsilon)
        return normalized_linear_output * gamma + beta

    # 若is_training为真，返回batch_norm_training，反之返回batch_norm_inference
    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)


def conv_layer(prev_layer, layer_depth, is_training):
    """
    Create a convolutional layer with the given layer as input.

    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param layer_depth: int
        We'll set the strides and number of feature maps based on the layer's depth in the network.
        This is *not* a good way to make a CNN, but it helps us create this example with very little code.
    :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
    :returns Tensor
        A new convolutional layer
    """
    strides = 2 if layer_depth % 3 == 0 else 1

    # 获取传进来的图片的 第4维度数量（即图片的depth）
    in_channels = prev_layer.get_shape().as_list()[3]
    out_channels = layer_depth * 4

    weights = tf.Variable(tf.truncated_normal([3, 3, in_channels, out_channels], stddev=0.05))

    layer = tf.nn.conv2d(prev_layer, weights, strides=[1, strides, strides, 1], padding='SAME')

    gamma = tf.Variable(initial_value=tf.ones([out_channels]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([out_channels]), trainable=True)

    pop_mean = tf.Variable(initial_value=tf.zeros([out_channels]), trainable=False)
    pop_variance = tf.Variable(initial_value=tf.ones([out_channels]), trainable=False)

    epsilon = 1e-3  # 防止除以0的。

    def batch_norm_training():
        # 1、tf.nn.moments 计算`x`的均值和方差.
        batch_mean, batch_variance = tf.nn.moments(layer, axes=[0, 1, 2], keep_dims=False)

        # 2、构建更新 全局统计量的assign赋值操作符
        decay = 0.99
        train_mean = tf.assign(
            ref=pop_mean, value=pop_mean * decay + (1 - decay) * batch_mean
        )
        train_variance = tf.assign(
            pop_variance, value=pop_variance * decay + (1 - decay) * batch_variance
        )
        # 3、构建控制依赖项，并最终执行当前批次的批归一化操作。
        with tf.control_dependencies(control_inputs=[train_mean, train_variance]):
            # 执行当前批量数据的归一化
            normalized_linear_output = (layer - batch_mean) / tf.sqrt(batch_variance + epsilon)
            return normalized_linear_output * gamma + beta

    def batch_norm_inference():
        normalized_linear_output = (layer - pop_mean) / tf.sqrt(pop_variance + epsilon)
        return normalized_linear_output * gamma + beta

    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return tf.nn.relu(batch_normalized_output)





def train_BN1(num_batches, batch_size, learning_rate):
    # Build placeholders for the input samples and labels
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    # Add placeholder to indicate whether or not we're training the model
    is_training = tf.placeholder(tf.bool)

    # todo - 做了一个循环，建了20个卷积层 Feed the inputs into a series of 20 convolutional layers
    layer = inputs
    for layer_i in range(1, 20):
        layer = conv_layer(layer, layer_i, is_training)

    # Flatten the output from the convolutional layers
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

    # Add one fully connected layer
    layer = fully_connected(layer, 100, is_training)

    # 获得分对数  Create the output layer with 1 node for each
    logits = tf.layers.dense(layer, 10)

    # Define loss and training operations
    model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(model_loss)

    # Create operations to test accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train and test the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch_i in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # train this batch
            sess.run(train_opt, {inputs: batch_xs, labels: batch_ys, is_training: True})

            # Periodically check the validation or training loss and accuracy
            if batch_i % 100 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: mnist.validation.images,
                                                              labels: mnist.validation.labels,
                                                              is_training: False})
                print(
                    'Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(batch_i, loss, acc))
            elif batch_i % 25 == 0:
                loss, acc = sess.run([model_loss, accuracy], {inputs: batch_xs, labels: batch_ys, is_training: False})
                print('Batch: {:>2}: Training loss: {:>3.5f}, Training accuracy: {:>3.5f}'.format(batch_i, loss, acc))

        # At the end, score the final accuracy for both the validation and test sets
        acc = sess.run(accuracy, {inputs: mnist.validation.images,
                                  labels: mnist.validation.labels,
                                  is_training: False})
        print('Final validation accuracy: {:>3.5f}'.format(acc))
        acc = sess.run(accuracy, {inputs: mnist.test.images,
                                  labels: mnist.test.labels,
                                  is_training: False})
        print('Final test accuracy: {:>3.5f}'.format(acc))

        # Score the first 100 test images individually, just to make sure batch normalization really worked
        correct = 0
        for i in range(100):
            correct += sess.run(accuracy, feed_dict={inputs: [mnist.test.images[i]],
                                                     labels: [mnist.test.labels[i]],
                                                     is_training: False})

        print("Accuracy on 100 samples:", correct / 100)

num_batches = 800
batch_size = 64
learning_rate = 0.002

tf.reset_default_graph()
with tf.Graph().as_default():
    train_BN1(num_batches, batch_size, learning_rate)

# todo-注意到，在我们最开始的几百个batches中，哪怕使用BN但准确率并不高。也就是说BN在最开始的训练中并没有起作用，
# 你需要给你的网络一点时间去学习。