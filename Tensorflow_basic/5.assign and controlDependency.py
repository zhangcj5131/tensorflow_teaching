import tensorflow as tf

def test_assign():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [], 'x')
        n = tf.Variable(initial_value=0., dtype = tf.float32, name = 'n')

        # n = n + x
        n_assign = tf.assign(n, n + x)
        with tf.control_dependencies([n_assign]):
            y = n*1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.summary.FileWriter('./log/5', graph=sess.graph)
            for i in [1,2,3,4]:
                n_ = sess.run(y, feed_dict={x:i})
                print(n_)

if __name__ == '__main__':
    test_assign()