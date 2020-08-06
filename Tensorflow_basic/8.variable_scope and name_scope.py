import tensorflow as tf


def f1():
    w = tf.Variable(initial_value=tf.random_normal(shape=[2], mean=0., stddev=1.),
                    dtype=tf.float32, name = 'w', trainable=False)
    return w

def f2(initializer=tf.random_normal_initializer):
    w = tf.get_variable(name='w', shape=[2], initializer=initializer,dtype = tf.float32)
    return w

def h1():
    w11 = f1()
    w12 = f1()
    print('w11=%s, w12=%s' % (w11.name, w12.name))

    w21 = f2()
    tf.get_variable_scope().reuse_variables()
    w22 = f2()
    print('w21=%s, w22=%s' % (w21.name, w22.name))

def h2():
    with tf.name_scope('name'), tf.variable_scope('variable'):
        w11 = f1()
        w12 = f1()
        print('w11=%s, w12=%s' % (w11.name, w12.name))

    with tf.name_scope('name'):
        with tf.variable_scope('variable', initializer=tf.random_normal_initializer):
            w21 = f2(initializer=None)
            tf.get_variable_scope().reuse_variables()
            w22 = f2()
            print('w21=%s, w22=%s' % (w21.name, w22.name))


    vars = tf.trainable_variables()
    print(len(vars))
    var_list = [var for var in vars if var.name.startswith('name')]
    print(len(var_list), var_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        temp = tf.get_default_graph().get_tensor_by_name('variable/w:0')
        print(sess.run(temp))



if __name__ == '__main__':
    h2()