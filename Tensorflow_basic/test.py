

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
模拟一元线性回归数据 ，请大家用numpy自己生成。
"""

if __name__ == '__main__':
    with tf.Graph().as_default():
        # 一、构建模型图
        # 1、构建输入的占位符
        input_x = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name='input_x'
        )
        input_y = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name='input_x'
        )

        # 2、构建变量
        w = tf.get_variable(
            name='w', shape=[1, 1], dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.1)
        )
        b = tf.get_variable(
            name='b', shape=[1], dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        # 3、正向传播，得到预测值。
        y_pred = tf.matmul(input_x, w) + b
        # 4、计算模型损失(MSE)
        loss = tf.reduce_mean(tf.square(input_y - y_pred))

        # 5、定义优化器。(含义：使用梯度下降的方式求解让损失函数最小的模型参数)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_opt = optimizer.minimize(loss=loss)

        # 二、构建会话
        with tf.Session() as sess:
            # a、变量初始化
            sess.run(tf.global_variables_initializer())
            # b、加载数据（训练的数据生成）
            N = 100
            train_x = np.linspace(0, 6, N) + np.random.normal(0, 2.0, N)
            train_y = train_x * 14 + 7 + np.random.normal(0, 5.0, N)
            train_x.shape = -1, 1
            train_y.shape = -1, 1
            print(train_x.shape, train_y.shape)
            # c、模型训练
            for e in range(1, 200):
                # 执行模型训练操作
                feed = {input_x: train_x, input_y: train_y}
                _, train_loss = sess.run([train_opt, loss], feed_dict=feed)
                print('Epoch:{} - Train Loss:{:.5f}'.format(e, train_loss))

            # d、使用训练数据，得到该数据的预测值，并做一个可视化操作。
            predict = sess.run(y_pred, feed_dict={input_x: train_x})
            plt.plot(train_x, train_y, 'ro')
            plt.plot(train_x, predict, 'g-')
            plt.show()
