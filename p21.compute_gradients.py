import tensorflow as tf

with tf.Graph().as_default():
    x = tf.get_variable('x', [], dtype = tf.float32, initializer=tf.initializers.constant(3.))
    w = tf.get_variable('w', [], dtype=tf.float32, initializer=tf.initializers.constant(4.))

    loss = w*x

    grad_w = tf.gradients(loss, [w])
    grad_x = tf.gradients(loss, [x])

    opt = tf.train.AdamOptimizer(0.1)
    grads_vals = opt.compute_gradients(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g_w = sess.run(grad_w)
        print('g_w=', g_w)

        g_x = sess.run(grad_x)
        print('g_x=', g_x)

        grads_values = sess.run(grads_vals)
        print('grads_vals=', grads_values)

