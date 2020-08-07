import tensorflow as tf


def scalar():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [], name = 'x')
        n = tf.get_variable('n', [], tf.float32, tf.zeros_initializer)
        tf.summary.scalar('n', n)

        n_assign = tf.assign(n , n + x)
        with tf.control_dependencies([n_assign]):
            y = n * 2
            tf.summary.scalar('y', y)

        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(logdir='./models/9', graph=sess.graph)
            step = 1
            for i in [1,2,3,4,5]:
                summary_op_, y_ = sess.run([summary_op, y], feed_dict={x: i})
                writer.add_summary(summary_op_, step)
                step += 1
                print(y_)
            writer.close()

if __name__ == '__main__':
    scalar()