import tensorflow as tf
'''
对于control_dependencies这个管理器，只有当里面的操作是一个op时，才会生效，也就是先执行传入的参数op，再执行里面的op。
而y=x仅仅是tensor的一个简单赋值，不是定义的op，所以在图中不会形成一个节点，这样该管理器就失效了。
tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中，这时control_dependencies就会生效，所以第二种情况的输出符合预期。
'''
x = tf.Variable(0.0)
y = tf.Variable(0.0)

#返回一个op，表示给变量x加1的操作
# x_plus_1 = tf.assign_add(x, 1)
x_plus_1 = tf.assign(x, x+1)
#control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前，
#先执行control_dependencies参数中的内容（在这里就是 x_plus_1），这里的解释不准确，先接着看。。。
with tf.control_dependencies([x_plus_1]):
    # y = x
    y = tf.identity(x)



with tf.Session() as session:
    init = tf.initialize_all_variables()
    init.run()
    for i in range(5):
        print(y.eval())#相当于sess.run(y)，按照我们的预期，由于control_dependencies的作用，所以应该执行print前都会先执行x_plus_1，但是这种情况会出问题