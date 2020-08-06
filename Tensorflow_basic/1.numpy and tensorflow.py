import numpy as np
import tensorflow as tf

x = np.random.rand(100)
y = x * 0.2 + 0.3
print(x.shape, y.shape)

with tf.Graph().as_default():
    w = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=-1, maxval=1))
    b = tf.Variable(initial_value=tf.zeros(shape=[]))
    y_pred = x * w + b
    loss = tf.reduce_mean(tf.square(y - y_pred))

    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    trian_op = opt.minimize(loss=loss)

    print(w, b, y_pred, loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(1, 801):
            _, lo = sess.run([trian_op, loss])
            if e % 20 == 0:
                print('epoch=%d, loss=%f' % (e, lo))
                print(sess.run(w), sess.run(b))

