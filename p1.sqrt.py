import tensorflow as tf

def my_sqrt(value, lr = 0.001, epoches = 12000):
    #x = tf.Variable()
    x = tf.get_variable(name = 'x', shape = [], dtype = tf.float32)
    loss = tf.square(x**2 - value)
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            _, lo = sess.run([train_op, loss])
            print('epoch=%d, loss = %f' % (epoch, lo))
        result = sess.run(x)
        return result






if __name__ == '__main__':
    value = 100
    print('sqrt(%f) = %f' % (value, my_sqrt(value)))