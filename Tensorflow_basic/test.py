
import tensorflow as tf

"""
tf.summary.scalar(name, tensor, collections=None, family=None):
  输出一个 `Summary` 对象（包含的是单个标量值）
  Args:
    name: 生成的节点的名字.
    tensor: 单个实数值

tf.summary.image(name, tensor, max_outputs=3, collections=None, family=None):
  输出图片.
    `max_outputs` 输出图片的最大数量。
     `tensor` ：必须是 4-D的tenor ,shape = `[batch_size,height, width, channels]` 
               `channels` 如下:
                  *  1: `tensor` is interpreted as Grayscale.
                  *  3: `tensor` is interpreted as RGB.
                  *  4: `tensor` is interpreted as RGBA.

tf.summary.histogram(name, values, collections=None, family=None):
  生成直方图，将你的数据以直方图的方式进行可视化，了解数据分布。
     values:  实数的`Tensor`. 任意形状
"""


def tensorboard():
    """
    实现一个求解n阶乘值乘以3的需求（构建一个控制依赖项），使用tf.control_dependencies
    :return:
    """
    with tf.Graph().as_default():
        # 1、构建输入的占位符，表示一个数字。
        input_x = tf.placeholder(tf.float32, None, 'input_x')

        # 2、定一个变量，表示阶乘的值。
        sum_x = tf.Variable(
            initial_value=1.0, dtype=tf.float32, name='sumx'
        )
        # todo 加一段可视化代码
        tf.summary.scalar(name='sum_x', tensor=sum_x)

        # 3、做一个乘法操作
        temp = sum_x * input_x
        # 将temp这个tensor的值，再次的赋值给sum_x
        assign_opt = tf.assign(
            ref=sum_x, value=temp
        )
        # 4、做一个阶乘的累加值，再乘以3的操作。
        with tf.control_dependencies(control_inputs=[assign_opt]):
            # fixme 当前with语句块中的代码执行之前，一定会触发control_inputs中给定的tensor操作
            y = sum_x * 3
            # todo 加一段可视化代码
            tf.summary.scalar(name='y', tensor=y)

        # 二、执行会话
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # todo 合并所有的summary 可视化输出操作。如果你图中没有定义，则返回None。
            summary = tf.summary.merge_all()
            # 构建一个日志输出对象
            writer = tf.summary.FileWriter(
                logdir='./models/ai20', graph=sess.graph
            )

            print('sum_x更新之前的值为:{}'.format(sess.run(sum_x)))
            step = 1
            for data in range(1, 6):
                y_, summary_ = sess.run([y, summary], feed_dict={input_x: data})
                # 将可视化输出的相关信息写入到磁盘文件中
                writer.add_summary(summary_, global_step=step)
                step +=1
            print('sum_x更新之后的值为:{}'.format(sess.run(sum_x)))
            writer.close()


if __name__ == '__main__':
    tensorboard()