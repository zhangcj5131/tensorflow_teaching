import tensorflow as tf
import numpy as np

def activation_test():
    with tf.Graph().as_default():
        x = tf.constant(value = np.random.normal(size=[3,3]), dtype = tf.float32)
        w = tf.get_variable('w', shape=[3,2], dtype = tf.float32, initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', shape = [2], dtype=tf.float32, initializer=tf.zeros_initializer())
        output = tf.matmul(x, w) + b
        output1 = tf.nn.relu(output)
        output2 = tf.nn.relu6(output)
        output3 = tf.nn.leaky_relu(output)
        alpha = 0.2
        #这就是 leaky_relu
        output4 = tf.maximum(alpha*output, output)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output, output1, output2, output3, output4 = sess.run([output, output1, output2, output3, output4])
            print(output, '\n', output1, '\n', output2, '\n', output3, '\n', output4)


if __name__ == '__main__':
    activation_test()