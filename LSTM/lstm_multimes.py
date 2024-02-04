"""
作者：赵江同
日期：2024年01月28日，21时：53分
"""
# coding=utf-8
from pandas import read_csv
from datetime import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import numpy as np
import pandas as pd


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
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# 缩放
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
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
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# 计算MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 读取CSV文件数据
data = pd.read_csv('D:\Program Files\JetBrains\pythonProject\huaweiyuce/alldata.csv', header=1, encoding='latin1')
# 提取第一列的数据，并将其作为datetime列
datetime_column = pd.to_datetime(data.iloc[:, 0], errors='coerce')
# 创建包含'datetime'和'value'两列的DataFrame
series = pd.DataFrame({'datetime': datetime_column, 'value': data.iloc[:, 1]})
# 将datetime列设置为索引
series = series.set_index('datetime')

# 设定窗口大小
window_size = 190
n = len(series)

# 初始化变量
error_scores = []

# 以窗口大小为步长滑动
for i in range(0, n - window_size + 1, window_size):
    # 选取窗口内的数据
    window_data = series.iloc[i:i + window_size, :]

    # 让数据变成稳定的
    raw_values = window_data.values
    diff_values = difference(raw_values, 1)

    # 把稳定的数据变成有监督数据
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # 数据拆分
    split_ratio = 0.8
    split_idx = int(len(supervised_values) * split_ratio)
    train, test = supervised_values[:split_idx], supervised_values[split_idx:]

    # 数据缩放
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit 模型
    lstm_model = fit_lstm(train_scaled, 1, 20, 50)

    # 测试数据的前向验证
    predictions = []
    n_test = len(test_scaled)
    for j in range(n_test):
        X, y = test_scaled[j, 0:-1], test_scaled[j, -1]
        X = X.reshape(1, 1, len(X))
        yhat = lstm_model.predict(X, batch_size=1)
        # 逆缩放和逆差分
        yhat = invert_scale(scaler, X[0, 0, :], yhat)
        yhat = inverse_difference(raw_values, yhat, n_test + 1 - j)
        predictions.append(yhat)

    # 计算指标
    y_test = raw_values[-len(test):, 0]
    predicted_prices = np.array(predictions)
    mape = calculate_mape(y_test, predicted_prices)
    mse = mean_squared_error(y_test, predicted_prices)
    r2 = r2_score(y_test, predicted_prices)

    # 输出指标
    print(f"Window {i + 1}: MAPE={mape}, MSE={mse}, R-squared={r2}")

    # 将指标加入列表
    error_scores.append((mape, mse, r2))

# 打印每一轮的MAPE
for i, (mape, _, _) in enumerate(error_scores):
    print(f"MAPE for Window {i + 1}: {mape}")

# 打印平均指标
avg_mape = np.mean([score[0] for score in error_scores])
avg_mse = np.mean([score[1] for score in error_scores])
avg_r2 = np.mean([score[2] for score in error_scores])
print(f"Average MAPE={avg_mape}, Average MSE={avg_mse}, Average R-squared={avg_r2}")
