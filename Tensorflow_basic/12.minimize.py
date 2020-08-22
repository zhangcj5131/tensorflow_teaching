import tensorflow as tf

def f1():
    with tf.Graph().as_default():
        with tf.variable_scope('network'):
            #1,3
            x = tf.constant(value = [[1,3,5]], dtype=tf.float32, name='x')
            #1,1
            y = tf.constant(value=[[5]], dtype=tf.float32, name='y')

            w = tf.Variable(initial_value=[[1],[1],[1]], dtype=tf.float32, name = 'w')
            b = tf.Variable(initial_value=[0], dtype=tf.float32, name='b')

            y_pred = tf.matmul(x, w) + b
            loss = tf.reduce_mean(tf.square(y - y_pred))
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            # train_op = optimizer.minimize(loss)
        with tf.variable_scope('optimizer'):
            '''
            grads_vars=[(gradient, variable),(gradient, variable),------]
            '''
            grads_vars = optimizer.compute_gradients(loss, var_list=[w,b])
            #梯度裁剪,之能用于解决梯度爆炸
            grads_vars_new = grads_vars.copy()
            for i, (gradient, variable) in enumerate(grads_vars_new):
                grads_vars_new[i] = (tf.clip_by_value(gradient, -5, 5),variable)


            train_op = optimizer.apply_gradients(grads_vars_new)
            #grads = tf.gradients(ys=loss, xs=[w,b])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('before train, [w, b]={}'.format(sess.run([w,b])))
            print('grads_vars={}'.format(sess.run(grads_vars_new)))
            # print('grads={}'.format(sess.run(grads)))
            sess.run(train_op)
            print('after train, [w, b]={}'.format(sess.run([w, b])))

if __name__ == '__main__':
    f1()