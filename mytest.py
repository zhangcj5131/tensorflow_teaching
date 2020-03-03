import tensorflow as tf

# w = tf.Variable(initial_value=10., dtype='float32')
# y = w*w + 2
#
# opt = tf.train.GradientDescentOptimizer(0.1)
# '''
# y=w^2 + 2 ,则y’=2w,w的初始值为10，因此y’(10)=20
# 所以computer_gradients（）函数返回的元组的第一个值表示y对w求导数的结果，第二个值表示w的值
# '''
# grad = opt.compute_gradients(y, [w])
#
# #而tf.gradients()则只会返回计算得到的梯度，而不会返回对应的variable。
# grad1 = tf.gradients(y, [w])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(grad))
#     print(sess.run(grad1))



# x = tf.Variable(initial_value=50., dtype='float32')
# w = tf.Variable(initial_value=10., dtype='float32')
# y = w*x
#
# opt = tf.train.GradientDescentOptimizer(0.1)
# grad = opt.compute_gradients(y, [w,x])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(grad))




with tf.Graph().as_default():
    x = tf.Variable(initial_value=3., dtype='float32')
    w = tf.Variable(initial_value=4., dtype='float32')
    y = w * x

    grad_w = tf.gradients(y, [w])
    grad_x = tf.gradients(y, [x])


    opt = tf.train.GradientDescentOptimizer(0.1)
    grads_vals = opt.compute_gradients(y, [w])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(grad_w))
        print(sess.run(grad_x))
        print(sess.run(grads_vals))




