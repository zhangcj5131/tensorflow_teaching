import numpy as np
import pandas as pd

admissions = pd.read_csv('./data/admissions.csv')

def explor_data(admissions):
    print(admissions.head())
    print(admissions.info())
    print(admissions.describe())
    print(admissions['admit'].value_counts())

def data_transform(admissions):
    data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis = 1)
    data = data.drop('rank', axis = 1)
    for field in ['gre', 'gpa']:
        mean, std = data[field].mean(), data[field].std()
        data.loc[:, field] = (data.loc[:, field] - mean) / std
    np.random.seed(42)
    sample_index = np.random.choice(data.index, int(len(data) * 0.9), replace=False)
    data_train, data_test = data.iloc[sample_index], data.drop(sample_index)
    x_train, y_train = data_train.drop('admit', axis = 1), data_train['admit']
    x_test, y_test = data_test.drop('admit', axis = 1), data_test['admit']
    return x_train, y_train, x_test, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bp_work(x_train, y_train, x_test, y_test, lr = 0.01, epoches = 2000, n_hidden = 2):
    n_records, n_features = x_train.shape

    last_loss  = None
    weights_input_hidden = np.random.normal(scale=1/n_features ** 0.5, size = [n_features, n_hidden])
    weights_hidden_output = np.random.normal(scale=1/n_hidden ** 0.5, size = n_hidden)

    for e in range(1, epoches):
        del_w_input_hidden = np.zeros(shape = weights_input_hidden.shape)
        del_w_hidden_output = np.zeros(shape = weights_hidden_output.shape)

        for x, y in zip(x_train.values, y_train):
            hidden_input = np.dot(x, weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            output = sigmoid(np.dot(hidden_output, weights_hidden_output))

            error = output - y
            error_term = error * output * (1 - output)
            hidden_error = error_term * weights_hidden_output
            hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

            del_w_hidden_output += error_term * hidden_output
            del_w_input_hidden += hidden_error_term * x[:, None]#6,2
            # print(hidden_error_term.shape)
            # print(x.shape)
            # print(x[:, None].shape)
        weights_input_hidden -= lr * del_w_input_hidden / n_records
        weights_hidden_output -= lr * del_w_hidden_output / n_records

        if e % 20 == 0:
            hidden = sigmoid(np.dot(x_train, weights_input_hidden))
            out = sigmoid(np.dot(hidden, weights_hidden_output))
            loss = np.mean((out - y_train) ** 2)
            if last_loss and loss > last_loss:
                print('loss is increasing!')
            else:
                print('train loss=', loss)
            last_loss = loss

    hidden = sigmoid(np.dot(x_test, weights_input_hidden))
    out = sigmoid(np.dot(hidden, weights_hidden_output))
    predict = out > 0.5
    acc = np.mean(predict == y_test)
    print(acc)




if __name__ == '__main__':
    #explor_data(admissions)
    x_train, y_train, x_test, y_test = data_transform(admissions)
    bp_work(x_train, y_train, x_test, y_test, lr = 0.01, epoches = 2000, n_hidden = 2)