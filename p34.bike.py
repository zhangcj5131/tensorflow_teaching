
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
项目提交邮箱：yingjun@ibeifeng.com
请注明：班级 + 名字。

要求提交：
     1、python源码；
     2、loss跑出来的画图（要求验证损失低于0.4）；
     3、最终在测试数据集上的拟合曲线。
"""
# 解决pd中print中间省略的问题
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 读取数据
data_path = './data/bike_sharing/hour.csv'
rides = pd.read_csv(data_path)


def f1():
    print(rides.head())
    print(rides.describe())
    rides.info()


def f2(rides):
    """
    # 一、季节、天气（分类变量）、月份、小时、星期几
    都是分类变量，需要调整为哑变量。
    """
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)
    """
    二、除了将 上面的原变量，还有以下变量需要删除，思考下why?
    1、instant 记录索引号；
    2、dteday  具体某天的日期号；
    3、atemp  体感温度，和temp重复，故删除；
    4、workingday 是否工作日，和weekday重复了，故删除；
    """
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    # print(data.head())

    rides[:24 * 10].plot(x='dteday', y='cnt')
    return data


def f3(data):
    """
    连续变量的数据标准化
    注意:cnt 就是target
    """
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # 将换算因子进行保存，以便在预测的时候还原数据。
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std
    return data, scaled_features


def f4(data):
    """
    拆分数据集，拆分 特征 和 target
    """
    # 保存最后21天 作为测试数据集
    test_data = data[-21 * 24:]

    # 移除最后21天数据，作为训练数据集
    data = data[:-21 * 24]

    # 将特征值 和 target进行拆分
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    return features, targets, test_features, test_targets


def f5(features , targets):
    """
    使用训练数据集的后60天数据，作为验证数据集；在模型训练过程中进行验证模型的效果。
    """
    train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
    val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]
    return train_features, train_targets, val_features, val_targets


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        :param input_nodes:   输入的节点数量 （特征数量）
        :param hidden_nodes:  隐藏层节点数量
        :param output_nodes:  输出层节点数量
        :param learning_rate:
        """
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 初始化权重
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        size=(self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         size=(self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # TODO: 设置 self.activation_function 来部署 sigmoid 函数
        self.activation_function = lambda x: 1/(1+np.exp(-x))

        # 如果你对 lambda表达式不熟悉，也可以用下面方式部署：
        # def sigmoid(x):
        #    return None
        # self.activation_function = sigmoid

    def train(self, features, targets):
        ''' 使用 batch==1 的features and targets训练网络
            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # 1、正向传播，None的地方都需要你编程

            # TODO: 隐藏层
            hidden_inputs = np.matmul(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            # TODO: 输出层
            # print(hidden_outputs.shape)
            final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs

            # 2、反向传播

            # TODO: Output error
            error = final_outputs - y  # 为预测值和真实值的差异

            # TODO: 计算隐藏层对误差error的贡献
            output_error_term = error * 1

            # TODO: 反向传播 error terms
            hidden_error = output_error_term * self.weights_hidden_to_output.reshape(self.hidden_nodes)


            hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)

            # 输入to隐藏层 权重梯度累加步骤 Weight step (input to hidden)
            delta_weights_i_h += X[:, None] * hidden_error_term
            # 隐藏层to输出层 权重梯度累加步骤Weight step (hidden to output)
            delta_weights_h_o += hidden_outputs[:, None] * output_error_term

        # TODO: 更新权重Update the weights
        self.weights_hidden_to_output -= delta_weights_h_o * self.lr / n_records
        self.weights_input_to_hidden -= delta_weights_i_h * self.lr / n_records

    def run(self, features):
        '''
        预测函数。
        使用输入特征，执行1次正向传播，得到预测值
        features: 1D array of feature values
        '''
        # 部署正向传播
        hidden_inputs = np.matmul(features.values, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: 输出层
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs


def MSE(y, Y):
    return np.mean((y-Y)**2)

# 显示训练过程中的训练 和 验证损失
def show(losses):
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()
    plt.show()


def test(network, scaled_features, test_features, test_targets, rides):
    fig, ax = plt.subplots(figsize=(8, 4))
    mean, std = scaled_features['cnt']
    predictions = network.run(test_features).T * std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt'] * std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_features.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)


if __name__ == '__main__':
    # f1()
    data = f2(rides)
    data, scaled_features = f3(data)
    # print(data.head())
    features, targets, test_features, test_targets = f4(data)

    train_features, train_targets, val_features, val_targets = f5(features, targets)
    print(train_features)
    # # todo 设置超参数 ###
    epochs = 2000        # 迭代次数
    learning_rate = 0.5  # 学习率
    hidden_nodes = 8   # 隐藏层节点数量，决定你模型的复杂度。
    output_nodes = 1   # 输出层的节点数量。

    n_features = train_features.shape[1]

    network = NeuralNetwork(n_features, hidden_nodes, output_nodes, learning_rate)

    losses = {'train': [], 'validation': []}
    for epoch in range(epochs):
        # 每次随机从训练数据集中抽取128条记录作为训练
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

        network.train(X, y)

        # 打印出训练过程
        train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
        if epoch % 80 == 0:
            print('训练迭代次数：{},训练损失:{} ,验证损失:'
                  '{}'.format(epoch, train_loss, val_loss))

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    show(losses)
    test(network, scaled_features, test_features, test_targets, rides)
