

"""
案例：研究生学院的录取数据，用梯度下降训练一个网络。
数据有3个特征：GRE  GPA  和 本科院校的排名 rank(从1到4)，1代表最好的，4代表最差的。
"""


import numpy as np
import pandas as pd


admissions = pd.read_csv(filepath_or_buffer='data/admissions.csv')


def data_explore(admissions):
    print(admissions.head(10))
    print(admissions.info())
    print(admissions.describe())
    print(admissions['admit'].value_counts())

"""
数据处理：
    1、分类变量进行哑编码。
    2、对连续变量标准化。
"""


def data_transform(admissions):
    # 1、分类变量哑编码
    data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
    data = data.drop('rank', axis=1)

    # 2、gre gpa的标准化（消除量纲）
    # fixme 注意标准做法：先拆分数据集，再用训练数据集的统计量去 标准化验证 和 测试数据集。
    for field in ['gre', 'gpa']:
        mean, std = data[field].mean(), data[field].std()
        data.loc[:, field] = (data[field] - mean) / std

    # print(data)
    # 3、数据集的拆分：（训练  和  测试数据集）
    np.random.seed(42)
    sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
    train_data, test_data = data.iloc[sample], data.drop(sample)

    # 4、特征和  目标值拆分（features 和 targets）
    features_train, targets_train = train_data.drop('admit', axis=1), train_data['admit']
    features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

    return features_train, targets_train, features_test, targets_test


# ***************************************
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# todo - 注意，虽然是分类任务，但我们这里是用的MSE作为损失函数
def gre_bp_work(features, targets, features_test, targets_test):
    # 超参数设置（这里我们只设置一层隐藏层）
    n_hidden = 2  # 为隐藏层节点数量，为 2
    epochs = 2000
    learnrate = 0.06

    # 获取 样本数量（n_records）  和特征数量(n_features)
    n_records, n_features = features.shape
    last_loss = None
    # 初始化权重
    # todo:注意有2个权重需要学习，所以均需要初始化
    weights_input_hidden = np.random.normal(
        scale=1 / n_features ** .5, size=(n_features, n_hidden)
    )
    weights_hidden_output = np.random.normal(
        scale=1 / n_hidden ** .5, size=n_hidden
    )

    for e in range(1, epochs):
        # 定义2个 del_w 初始化为0，用于存储梯度
        del_w_input_hidden = np.zeros(weights_input_hidden.shape)
        del_w_hidden_output = np.zeros(weights_hidden_output.shape)
        # 同样，我们每次遍历一个数据样本
        for x, y in zip(features.values, targets):
            ## 正向传播 ##
            # 需要编程: 计算 输出output
            hidden_input = np.matmul(x, weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            output = sigmoid(np.matmul(hidden_output, weights_hidden_output))

            ## 反向传播 ##
            # 需要编程: 计算预测误差
            error = output - y  # 标量

            # 需要编程: 对输出层output计算error term
            output_error_term = error * output * (1-output)  # 标量

            # 将误差反向传播给隐藏层
            # 需要编程: 计算隐藏层对 输出层误差的贡献
            hidden_error = output_error_term * weights_hidden_output  # 向量  (n_hidden, )

            # 需要编程: 计算隐藏层的error term
            hidden_error_term = hidden_error * hidden_output * (1-hidden_output)


            # 需要编程: 更新权重梯度  Update the change in weights
            del_w_hidden_output += hidden_output * output_error_term  # 向量 (n_hidden, )
            del_w_input_hidden += x[:, None] * hidden_error_term



            # 需要编程: 更新权重  Update weights
        weights_input_hidden -= del_w_input_hidden * learnrate / n_records
        weights_hidden_output -= del_w_hidden_output * learnrate / n_records

        # 打印均方差
        if e % 20 == 0:
            # 正向传播
            hidden_output = sigmoid(np.dot(features, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output,
                                 weights_hidden_output))
            loss = np.mean((out - targets) ** 2)

            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss

    # 计算准确率
    hidden = sigmoid(np.dot(features_test, weights_input_hidden))
    out = sigmoid(np.dot(hidden, weights_hidden_output))
    predictions = out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))


# todo - 注意，虽然是分类任务，但我们这里是用的MSE作为损失函数
def gre_bp_with_SGD(features, targets, features_test, targets_test):
    # 超参数设置（这里我们只设置一层隐藏层）
    n_hidden = 6  # 为隐藏层节点数量，为 2
    epochs = 2000
    learnrate = 0.06

    # 获取 样本数量（n_records）  和特征数量(n_features)
    n_records, n_features = features.shape
    last_loss = None
    # 初始化权重
    # todo:注意有2个权重需要学习，所以均需要初始化
    weights_input_hidden = np.random.normal(
        scale=1 / n_features ** .5, size=(n_features, n_hidden)
    )
    weights_hidden_output = np.random.normal(
        scale=1 / n_hidden ** .5, size=n_hidden
    )

    for e in range(1, epochs):
        # 定义2个 del_w 初始化为0，用于存储梯度
        # del_w_input_hidden = np.zeros(weights_input_hidden.shape)
        # del_w_hidden_output = np.zeros(weights_hidden_output.shape)
        # 同样，我们每次遍历一个数据样本
        for x, y in zip(features.values, targets):
            ## 正向传播 ##
            # 需要编程: 计算 输出output
            hidden_input = np.matmul(x, weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            output = sigmoid(np.matmul(hidden_output, weights_hidden_output))

            ## 反向传播 ##
            # 需要编程: 计算预测误差
            error = output - y  # 标量

            # 需要编程: 对输出层output计算error term
            output_error_term = error * output * (1-output)  # 标量

            # 将误差反向传播给隐藏层
            # 需要编程: 计算隐藏层对 输出层误差的贡献
            hidden_error = output_error_term * weights_hidden_output  # 向量  (n_hidden, )

            # 需要编程: 计算隐藏层的error term
            hidden_error_term = hidden_error * hidden_output * (1-hidden_output)
            # 需要编程: 更新权重梯度  Update the change in weights
            # 需要编程: 更新权重  Update weights
            weights_input_hidden -= x[:, None] * hidden_error_term * learnrate
            weights_hidden_output -= hidden_output * output_error_term * learnrate

        # 打印均方差
        if e % 20 == 0:
            # 正向传播
            hidden_output = sigmoid(np.dot(features, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output,
                                 weights_hidden_output))
            loss = np.mean((out - targets) ** 2)

            if last_loss and last_loss < loss:
                print('Epoch:{}'.format(e))
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print('Epoch:{}'.format(e))
                print("Train loss: ", loss)
            last_loss = loss

    # 计算准确率
    hidden = sigmoid(np.dot(features_test, weights_input_hidden))
    out = sigmoid(np.dot(hidden, weights_hidden_output))
    predictions = out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))



def get_batches(features_train, targets_train, batch_size=32):
    """
    构建获取小批量数据的生成器。
    :param features_train:
    :param targets_train:
    :param batch_size:  批量的大小
    :return:
    """
    # 断言，确保 特征的数量 和标签的数量一致。
    assert len(features_train) == len(targets_train)

    for i in range(0, len(features_train), batch_size):
        batch_x = features_train[i: i+batch_size]
        batch_y = targets_train[i: i+batch_size]
        # 利用yield 构造一个生成器。
        yield batch_x, batch_y


def gre_bp_with_Batch(features_train, targets_train, features_test, targets_test, batch_size=64):
    """
    gre反向传播,实现 MBGD（小批量梯度下降）
    :return:
    """
    # 1、设置超参数。
    n_hidden = 5  # 隐藏层节点数量
    epochs = 1000  # 迭代的次数
    learning_rate = 0.008

    # 2、获取样本的数量 和特征数量。
    n_records, n_features = features_train.shape
    last_loss = None
    # 3、构建模型权重。(有2个权重)
    weights_input_hidden = np.random.normal(
        loc=0.0, scale=1/n_features**0.5, size=(n_features, n_hidden)
    )
    weights_hidden_output = np.random.normal(
        scale=1/n_hidden**0.5, size=n_hidden
    )

    # 4、构建模型循环
    for e in range(1, epochs):
        # 构建遍历数据集的循环。
        for batch_x, batch_y in get_batches(features_train.values, targets_train.values, batch_size=batch_size):
            # 5、正向传播
            hidden_input = np.matmul(batch_x, weights_input_hidden)
            hidden_output = sigmoid(hidden_input)

            output = sigmoid(np.matmul(hidden_output, weights_hidden_output))  # (batch_size, ) 向量
            # 6、反向传播
            error = output - batch_y.reshape(-1)   # 向量

            output_error_term = error * output * (1-output)  # 向量

            hidden_error = output_error_term[:, None] * weights_hidden_output
            hidden_error_term = hidden_error * hidden_output * (1-hidden_output)

            # 7、执行梯度下降，更新权重。
            weights_hidden_output -= np.matmul(hidden_output.transpose(), output_error_term) * learning_rate / batch_size
            weights_input_hidden -= np.matmul(batch_x.transpose(), hidden_error_term) * learning_rate / batch_size

        # 8、每10个迭代，打印1次模型损失。
        if e % 20 == 0:
            hidden_output = sigmoid(np.dot(features_train, weights_input_hidden))
            train_pred = sigmoid(np.dot(hidden_output, weights_hidden_output))  # out就是预测值
            loss = np.mean((train_pred - targets_train)**2)

            if last_loss and last_loss < loss:
                print('警告，模型损失在上升')
            else:
                print('Epoch:{} - Train Loss:{}'.format(e, loss))
            last_loss = loss
    # 9、计算测试集的准确率。
    hidden = sigmoid(np.matmul(features_test.values, weights_input_hidden))
    out = sigmoid(np.matmul(hidden, weights_hidden_output))  # 测试数据集的预测值
    predictions = out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print('Test ACC:{}'.format(accuracy))


if __name__ == '__main__':
    feature_train, target_train, feature_test, target_test = data_transform(admissions)
    gre_bp_work(feature_train, target_train, feature_test, target_test)
    # print(feature_train)

    # features_train, targets_train, features_test, targets_test = data_transform(admissions)
    # gre_bp_work(features_train, targets_train, features_test, targets_test)
    # gre_bp_with_SGD(features_train, targets_train, features_test, targets_test)
    # features_train, targets_train, features_test, targets_test = data_transform(admissions)

    # for batch_x, batch_y in get_batches(features_train.values, targets_train, batch_size=64):
    #     print(batch_x.shape, batch_y.shape)
    # gre_bp_with_Batch(
    #     features_train, targets_train, features_test, targets_test, batch_size=128
    # )
    # gre_bp_answer(features_train, targets_train, features_test, targets_test)

