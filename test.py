import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决pd中print中间省略的问题
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data_path = './data/bike_sharing/hour.csv'
rides = pd.read_csv(data_path)

def data_explor(rides):
    print(rides.head())
    # print(rides.describe())
    # print(rides.info)

def data_tranform(rides):
    dummy_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
    for field in dummy_fields:
        dummy = pd.get_dummies(rides[field], prefix=field, drop_first=False)
        rides = pd.concat([rides, dummy], axis = 1)
    drop_fields = ['instant', 'dteday', 'season', 'mnth', 'hr', 'workingday', 'weekday', 'weathersit', 'atemp']
    rides = rides.drop(drop_fields, axis = 1)

    quant_fields = ['temp', 'hum', 'casual', 'registered', 'cnt', 'windspeed']
    scaled_features = {}
    for field in quant_fields:
        mean, std = rides[field].mean(), rides[field].std()
        scaled_features[field] = (mean, std)
        rides[field] = (rides[field] - mean) / std
    return rides, scaled_features

def train_test_split(data):
    train = data[:-21*24]
    test = data[-21*24:]
    target_fields = ['casual', 'registered', 'cnt']
    train_x, train_y = train.drop(target_fields, axis = 1), train[target_fields]
    test_x, test_y = test.drop(target_fields, axis = 1), test[target_fields]
    return train_x, train_y, test_x, test_y

def train_valid_split(train_x, train_y):
    train_feature, train_target = train_x[:-60*24], train_y[:-60*24]
    valid_feature, valid_target = train_x[-60*24:], train_y[-60*24:]
    return train_feature, train_target, valid_feature, valid_target


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        self.weights_input_hidden = np.random.normal(0.0, input_nodes ** -0.5, size=[input_nodes, hidden_nodes])
        self.weights_hidden_output = np.random.normal(0.0, hidden_nodes ** -0.5, size = [hidden_nodes, output_nodes])
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        n_records = features.shape[0]
        del_weights_input_hidden = np.zeros(shape=self.weights_input_hidden.shape)
        del_weights_hidden_output = np.zeros(shape = self.weights_hidden_output)

        for x, y in zip(features, targets):
            hidden_input = np.matmul(x, self.weights_input_hidden)
            hidden_output = self.activation_function(hidden_input)

            final_input = np.matmul(hidden_output, self.weights_hidden_output)
            final_output = final_input


if __name__ == '__main__':
    # data_explor(rides)
    data, scale_features = data_tranform(rides)
    features, targets, test_features, test_targets = train_test_split(data)
    train_features, train_targets, val_features, val_targets = train_valid_split(features, targets)

    print(train_features)
