
'''
不用 batch normalization
训练效果会很糟糕
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True, reshape=False)

def conv_layer(layer, layer_i):
    strides = 2 if layer_i==3 else 1
    layer = tf.layers.conv2d(layer, layer_i*4, 3, strides, 'same', activation=tf.nn.relu)
    return layer

def fully_connected(layer, num_units):
    layer = tf.layers.dense(layer, num_units, activation=tf.nn.relu)
    return layer

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


if __name__ == '__main__':
    num_batches = 800
    batch_size = 64
    learning_rate = 0.002

    tf.reset_default_graph()
    with tf.Graph().as_default():
        train(num_batches, batch_size, learning_rate)