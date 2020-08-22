

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 1], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y')
w = tf.Variable(initial_value=tf.constant(0.0))

# 定义一个计算步数的变量
global_steps = tf.Variable(initial_value=0, trainable=False)
# global_steps = tf.train.get_or_create_global_step()

learning_rate = tf.train.exponential_decay(
    learning_rate=0.1,
    global_step=global_steps,
    decay_steps=10,
    decay_rate=0.9
)

# 定义损失
loss = tf.pow(w*x - y, 2)

train_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).\
minimize(loss, global_step=global_steps)

# 构建会话
epochs = 3
batch_cnt = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch in range(batch_cnt):
            feed = {x: np.linspace(1, 2, 10).reshape([10, 1]),
                    y: np.linspace(1, 2, 10).reshape([10, 1])}
            sess.run(train_opt, feed)
            print(sess.run(learning_rate))
            print(sess.run(global_steps))