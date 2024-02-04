"""
作者：赵江同
日期：2024年02月01日，12时：33分
"""
"""
作者：赵江同
日期：2024年01月28日，19时：35分
"""
"""
作者：赵江同
日期：2024年01月23日，12时：32分
"""
# coding=utf-8
from pandas import read_csv
from datetime import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
# 读取时间数据的格式化
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# 转换成有监督数据
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df



# 转换成差分数据
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# 逆差分
def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]


# 缩放
def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # 转换test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# 逆缩放
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row, dtype=object)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]



# fit LSTM来训练数据
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))  # 输出层1个node
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # 按照batch_size，一次读取batch_size个数据
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print("当前计算次数："+str(i))
    return model


# # 1步长预测
# def forcast_lstm(model, batch_size, X):
#     X = X.reshape(1, 1, len(X))
#     yhat = model.predict(X, batch_size=batch_size)
#     return yhat[0, 0]


# 加载数据# 从CSV文件读取数据
data = pd.read_csv('D:\Program Files\JetBrains\pythonProject\huaweiyuce/alldata.csv', header=1, encoding='latin1')
# 提取第一列的数据，并将其作为datetime列
datetime_column = pd.to_datetime(data.iloc[:, 0], errors='coerce')
# 创建包含'datetime'和'value'两列的DataFrame
series_1 = pd.DataFrame({'datetime': datetime_column, 'value': data.iloc[:, 1]})
# 将datetime列设置为索引
series= series_1.set_index('datetime')
series[series == 0] = pd.NA
# 使用前向填充非空值（ffill）方法填充 NaN 值
series = series.ffill()
# 让数据变成稳定的
raw_values = series.values
diff_values = difference(raw_values, 1)#转换成差分数据

# 把稳定的数据变成有监督数据
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# 数据拆分：训练数据、测试数据，前24行是训练集，后12行是测试集

# 数据拆分：训练数据、测试数据，前24行是训练集，后12行是测试集
split_ratio = 0.8
split_idx = int(len(supervised_values) * split_ratio)

train, test = supervised_values[:split_idx], supervised_values[split_idx:]

# 数据缩放
scaler, train_scaled, test_scaled = scale(train, test)

#重复实验   
repeats = 1
error_scores = list()
for r in range(repeats):
    # fit 模型
    lstm_model = fit_lstm(train_scaled, 1, 20, 50)  # 训练数据，batch_size，epoche次数, 神经元个数
    # 预测
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)#训练数据集转换为可输入的矩阵
    lstm_model.predict(train_reshaped, batch_size=1)#用模型对训练数据矩阵进行预测
    # 测试数据的前向验证，实验发现，如果训练次数很少的话，模型回简单的把数据后移，以昨天的数据作为今天的预测值，当训练次数足够多的时候
    # 才会体现出来训练结果
    # 测试数据的前向验证
    predictions = list()
    n_test = len(test_scaled)
    for i in range(n_test):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        X = X.reshape(1, 1, len(X))  # 这里添加维度以匹配模型的输入形状
        yhat = lstm_model.predict(X, batch_size=1)
        # 逆缩放
        yhat = invert_scale(scaler, X[0, 0, :], yhat)
        # 逆差分
        yhat = inverse_difference(raw_values, yhat, n_test + 1 - i)
        predictions.append(yhat)
        expected_index = len(train) + i
        expected = raw_values[expected_index, 0]  # 假设你的实际值在每行的第一列
        print('第%d个时间步, 预测值=%f, 实际值=%f' % (expected_index + 1, yhat, expected))

    # 性能报告
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(raw_values[-n_test:], predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    error_scores.append(mse)
# 统计信息
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()

# 重新定义y_test和predicted_prices，以适应修改后的数据划分
y_test = raw_values[-len(test):]
predicted_prices = np.array(predictions)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predicted_prices)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test, predicted_prices)
print(f"R-squared (R2): {r2}")

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = calculate_mape(y_test, predicted_prices)
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# 绘制预测结果
plt.rc('font', family='SimHei')  # 用来正常显示中文标签
plt.plot(np.arange(len(y_test)), y_test, label='True Prices')  # 使用索引作为横坐标
plt.plot(np.arange(len(predicted_prices)), predicted_prices, label='Predicted Prices')  # 使用索引作为横坐标
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('cpu utilization')
plt.title('cpu utilization Prediction using LSTM')


# Plot True Prices, Predicted Prices, and Error Curve
absolute_errors = np.abs(y_test - predicted_prices)
plt.figure(figsize=(12, 6))

# Plot True Prices
plt.subplot(2, 1, 1)
plt.plot(y_test, label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('cpuPrice')
plt.title('cpu Price Prediction using LSTM')

# Plot Error Curve
plt.subplot(2, 1, 2)
plt.plot(absolute_errors, label='Absolute Errors', color='red')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Absolute Error')
plt.title('Absolute Errors over Time')

plt.tight_layout()
plt.show()
