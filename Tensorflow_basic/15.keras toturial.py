

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# 设置随机数
np.random.seed(42)

# 演示的数据
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]).astype('float32')
y = np.array([[0],
              [1],
              [1],
              [0]]).astype('float32')

# 对标签进行one-hot编码
y = np_utils.to_categorical(y)
print(y)

# 构建模型图
xor = Sequential()                # 构建一个线性的模型层级堆叠对象
"""
    Dense(self, units                       #  你要输出的节点数量
                 activation=None,           #  激活函数
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

# Example
    ```python
        # 如果作为第一层:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # 模型会自动将输入的 shape识别为 (*, 16)   即16是 特征数量
        # 输出的shape是 (*, 32)

        # 如果不是作为第一层，那么无需输入input_shape。他会自动识别上一层的节点数量。
        model.add(Dense(32))
    ```
"""
xor.add(Dense(64, input_dim=2))   # 第一层隐藏层 ，64代表该层隐藏层节点数量 (2*64)+64
xor.add(Activation("relu"))
xor.add(Dense(32))                # 第二层隐藏层，64*32+32
xor.add(Activation('relu'))
xor.add(Dense(2))                 # 输出层 32*2+2
xor.add(Activation("sigmoid"))

# 设置模型训练参数
"""
compile(self, optimizer,              用什么优化器
                loss=None,            损失函数
                metrics=None,         训练期间评估模型的指标。和损失函数类型，只是不会用于训练 (https://keras.io/zh/metrics/) ['accuracy', 'acc', 'crossentropy', 'ce']
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs):
"""
# loss  "categorical_crossentropy" 分类交叉熵    binary_crossentropy 二值交叉熵
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# #  打印模型的结构图，超级好用
print(xor.summary())

# 训练模型
"""
    fit(self,
            x=None,
            y=None,
            batch_size=None,              # 批次大小
            epochs=1,                     # 迭代次数
            verbose=1,                    # 显示训练进度模式 ： 0 = 不显示, 1 = progress bar, 2 = one line per epoch.
            callbacks=None,               # 钩子，早期停止技术。
            validation_split=0.,
            validation_data=None,         # 验证数据集 tuple `(x_val, y_val)`
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,         # 每1个epoch的步数，即 每一次迭代中 有多少个batch_size.等于 总样本数量/batch_size
            validation_steps=None,
            **kwargs):
"""
history = xor.fit(X, y, epochs=1000, verbose=1)

# 模型评估
score = xor.evaluate(X, y)
print("\n准确率是: ", score[-1])

# 打印预测值
print("\n预测值是:")
print(xor.predict_proba(X))