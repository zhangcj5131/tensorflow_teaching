import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True, reshape=False)

def fully_connected(layer, num_unists, is_training):
    layer = tf.layers.dense(layer, num_unists, use_bias=False, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def conv_layer(layer, layer_i, is_training):
    strides = 2 if layer_i%3 == 0 else 1
    layer = tf.layers.conv2d(layer, layer_i*4, 3, strides, 'same', use_bias=False)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer

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


if __name__ == '__main__':
    num_batches = 800
    batch_size = 64
    learning_rate = 0.002

    tf.reset_default_graph()
    with tf.Graph().as_default():
        train_BN(num_batches, batch_size, learning_rate)