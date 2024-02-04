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
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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


# 读取数据和预处理的部分保持不变
data = pd.read_csv('D:\Program Files\JetBrains\pythonProject\huaweiyuce/alldata.csv', header=1, encoding='latin1')
datetime_column = pd.to_datetime(data.iloc[:, 0], errors='coerce')
series_1 = pd.DataFrame({'datetime': datetime_column, 'value': data.iloc[:, 1]})
series = series_1.set_index('datetime')
series[series == 0] = pd.NA
# 使用前向填充非空值（ffill）方法填充 NaN 值
series = series.ffill()

#series_1 = series_1.set_index('datetime')
#chuangkou = 190
#series = series_1.iloc[:chuangkou, :]
raw_values = series.values
diff_values = difference(raw_values, 1)

supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# 使用所有数据进行拟合LSTM模型
scaler, train_scaled, _ = scale(supervised_values, supervised_values)  # 仅使用train数据进行scale

# 拟合模型的部分保持不变
lstm_model = fit_lstm(train_scaled, 1, 20, 50)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
# lstm_model.predict(train_reshaped, batch_size=1)

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
mape = calculate_mape(train_y, predicted_train_y)
#mape = calculate_mape(train_y[-38:], predicted_train_y[-38:])
print(f"Mean Absolute Percentage Error (MAPE) on Training Data: {mape}%")
mse = mean_squared_error(train_y, predicted_train_y)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squared (coefficient of determination)
r2 = r2_score(train_y, predicted_train_y)
print(f"R-squared (R2): {r2}")
# 绘制真实结果与预测结果的对比图
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(train_y, label='True Values')
plt.plot(predicted_train_y, label='Predicted Values')
plt.title('True vs Predicted Values on Training Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

# 绘制预测误差图
absolute_errors = np.abs(train_y - predicted_train_y)
plt.subplot(2, 1, 2)
plt.plot(absolute_errors, label='Absolute Errors', color='red')
plt.title('Absolute Errors on Training Data')
plt.xlabel('Time Step')
plt.ylabel('Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
