import numpy as np
import tensorflow as tf


def test_graph():
    print('default graph is:%s' % tf.get_default_graph())
    a = tf.constant(value = 1., dtype = tf.float32, shape = [], name = 'a')
    b = tf.constant(value = 2.)

    v1 = tf.add(x = a, y = np.random.random_sample(), name = 'v1')
    v2 = tf.add(x = b, y = tf.random_normal(shape = []), name = 'v2')

    result = tf.multiply(v1, v2, name = 'multiply')
    print(a, b, v1, v2, result)

def test_graph1():
    print('default graph is:%s' % tf.get_default_graph())
    a = tf.constant(value = 1., dtype = tf.float32, shape = [], name = 'a')
    b = tf.constant(value = 2.)


    # v1 = tf.add(x = a, y = np.random.random_sample(), name = 'v1')
    # v2 = tf.add(x = b, y = tf.random_normal(shape = []), name = 'v2')
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[])

    #result = tf.multiply(v1, v2, name = 'multiply')
    result = v1 * v2
    print(a, b, v1, v2, result)


def test_graph2():
    print('default graph is:%s' % tf.get_default_graph())
    a = tf.constant(value = 1., dtype = tf.float32, shape = [], name = 'a')
    with tf.Graph().as_default():
        print('default graph is:%s' % tf.get_default_graph())
        b = tf.constant(value = 2.)


    # v1 = tf.add(x = a, y = np.random.random_sample(), name = 'v1')
    # v2 = tf.add(x = b, y = tf.random_normal(shape = []), name = 'v2')
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[])

    #result = tf.multiply(v1, v2, name = 'multiply')
    result = v1 * v2
    print(a, b, v1, v2, result)



def test_session():
    print('default graph is:%s' % tf.get_default_graph())
    a = tf.constant(value = [1,2,3], dtype = tf.float32, shape = [3,5], name = 'a')
    b = tf.constant(value = [4,5,6,7], dtype = tf.float32, shape = [5,3])


    # v1 = tf.add(x = a, y = np.random.random_sample(), name = 'v1')
    # v2 = tf.add(x = b, y = tf.random_normal(shape = []), name = 'v2')
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[])

    #result = tf.multiply(v1, v2, name = 'multiply')
    result = tf.matmul(v1, v2)
    print(a, b, v1, v2, result)

    # sess = tf.Session()
    # # a = sess.run(a)
    # #     # result=sess.run(result)
    # a_, b_, v1_, v2_, result_ = sess.run(fetches=[a, b, v1, v2, result])
    # print(a_, b_, v1_, v2_, result_)
    #
    # sess.close()
    with tf.Session() as sess:
        a_, b_, v1_, v2_, result_ = sess.run(fetches=[a, b, v1, v2, result])
        print(a_, b_, v1_, v2_, result_)

def test_session1():
    print('default graph is:%s' % tf.get_default_graph())
    a = tf.constant(value = [1,2,3], dtype = tf.float32, shape = [3,5], name = 'a')
    b = tf.constant(value = [4,5,6,7], dtype = tf.float32, shape = [5,3])


    # v1 = tf.add(x = a, y = np.random.random_sample(), name = 'v1')
    # v2 = tf.add(x = b, y = tf.random_normal(shape = []), name = 'v2')
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[])

    #result = tf.multiply(v1, v2, name = 'multiply')
    result = tf.matmul(v1, v2)
    print(a, b, v1, v2, result)

    sess = tf.Session()
    print(a.eval(session = sess))
    print(b.eval(session=sess))


def test_session2():
    print('default graph is:%s' % tf.get_default_graph())
    a = tf.constant(value = [1,2,3], dtype = tf.float32, shape = [3,5], name = 'a')
    b = tf.constant(value = [4,5,6,7], dtype = tf.float32, shape = [5,3])


    # v1 = tf.add(x = a, y = np.random.random_sample(), name = 'v1')
    # v2 = tf.add(x = b, y = tf.random_normal(shape = []), name = 'v2')
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[])

    #result = tf.multiply(v1, v2, name = 'multiply')
    result = tf.matmul(v1, v2)
    print(a, b, v1, v2, result)

    # sess = tf.Session()
    #tf.InteractiveSession()自动作为默认 session,后面就可以不指定 session 了
    sess = tf.InteractiveSession()
    print(tf.get_default_session())
    print(a.eval())
    sess.close()



if __name__ == '__main__':
    test_graph()
    # print('---------------')
    # test_graph1()
    # print('---------------')
    # test_graph2()
    # test_session()
    # test_session2()