"""
作者：赵江同
日期：2024年02月01日，13时：37分
"""
"""
作者：赵江同
日期：2024年02月01日，12时：46分
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

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]

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

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row, dtype=object)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
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


# 读取数据和预处理的部分保持不变
data = pd.read_csv('D:\Program Files\JetBrains\pythonProject\huaweiyuce/alldata.csv', header=1, encoding='latin1')
datetime_column = pd.to_datetime(data.iloc[:, 0], errors='coerce')
series_1 = pd.DataFrame({'datetime': datetime_column, 'value': data.iloc[:, 1]})
# series = series_1.set_index('datetime')
# series[series == 0] = pd.NA
# # 使用前向填充非空值（ffill）方法填充 NaN 值
# series = series.ffill()

series_1 = series_1.set_index('datetime')
chuangkou = 190
series = series_1.iloc[:chuangkou, :]

raw_values = series.values
diff_values = difference(raw_values, 1)

supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# 使用所有数据进行拟合LSTM模型
scaler, train_scaled, _ = scale(supervised_values, supervised_values)  # 仅使用train数据进行scale

# 拟合模型的部分保持不变
lstm_model = fit_lstm(train_scaled, 1, 2, 50)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# 生成训练集的预测结果
train_predictions = list()
for i in range(len(train_scaled)):
    X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
    X = X.reshape(1, 1, len(X))
    yhat = lstm_model.predict(X, batch_size=1)
    yhat = invert_scale(scaler, X[0, 0, :], yhat)
    yhat = inverse_difference(raw_values, yhat, len(train_scaled) + 1 - i)
    train_predictions.append(yhat)

# 计算MAPE
train_y = raw_values[1:]  # 第一个值是无法预测的，因为我们做了差分
predicted_train_y = np.array(train_predictions)
import socket

SERVER_IP = '127.0.0.1'
SERVER_PORT = 8888

def send_data(data):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))

    # 将数据转换为字符串，并发送给服务器
    data_str = str(data)
    client_socket.send(data_str.encode())

    print(f"Data '{data}' sent successfully")

    client_socket.close()


individual_mapes = []
for i, (true_val, pred_val) in enumerate(zip(train_y, predicted_train_y)):
    mape_single = calculate_mape(true_val, pred_val)
    individual_mapes.append(mape_single)
    if mape_single > 1.0:
        combined_str = f"{data.iloc[i + 1, 0]}, {true_val}"
        send_data(combined_str)
#在1中我们已经实现了传输，，而且数据是对的，现在，我们只需要将日期对应上去就行了