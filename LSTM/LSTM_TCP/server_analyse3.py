"""
作者：赵江同
日期：2024年02月03日，12时：37分
"""
"""
作者：赵江同
日期：2024年02月03日，11时：53分
"""
"""
作者：赵江同
日期：2024年02月02日，21时：48分
"""
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
# 缩放

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
# 加载训练好的模型
lstm_model = load_model('lstm_model.h5')
data = pd.read_csv('D:\Program Files\JetBrains\pythonProject\LSTM\LSTM_TCP\data_received.csv', header=None, encoding='latin1')
# 对数值列进行处理，去掉方括号和空格并尝试转换为浮点数
datetime_column = pd.to_datetime(data.iloc[:, 0], errors='coerce')

# 创建包含'datetime'和'value'两列的DataFrame
series_1 = pd.DataFrame({'datetime': datetime_column, 'value': data.iloc[:, 1]})
# 将datetime列设置为索引
series_1 = series_1.set_index('datetime')

# 将数据重新采样成固定的时间间隔，这里假设是每5分钟一个数据点
resampled_data = series_1.resample('5T').mean()

def difference(data, interval=1):
    return np.diff(data, n=interval)

def scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = data.reshape(-1, 1)  # Reshape the data if it's 1D
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaler, scaled_data

def inverse_scale(scaler, scaled_data):
    return scaler.inverse_transform(scaled_data)

def inverse_difference(history, diff, interval=1):
    return diff + history[-interval]

def prepare_data(data, timesteps):
    diff_data = difference(data)
    scaler, scaled_data = scale(diff_data)
    X, y = [], []
    for i in range(len(scaled_data) - timesteps):
        X.append(scaled_data[i:i+timesteps])
        y.append(scaled_data[i+timesteps])
    X, y = np.array(X), np.array(y)
    return scaler, X, y

def make_prediction(model, X):
    return model.predict(X)

# 对缺失的数据点进行预测并填充
for i in range(len(resampled_data)):
    if np.isnan(resampled_data.iloc[i, 0]):
        scaler, X, y = prepare_data(data, timesteps=2)
        # Train your model here
        # Make sure you reshape X if needed for LSTM input shape
        # Make predictions
        predictions = make_prediction(lstm_model, X)
        # Inverse scale the predictions
        predictions = inverse_scale(scaler, predictions)
        # Inverse difference the predictions
        next_value = inverse_difference(data, predictions[-1])
        print("Next predicted value:", next_value)
        resampled_data.iloc[i, 0] = scaler.inverse_transform(y_pred)[0, 0]


# 将预测的数据逆缩放并填充到原始数据中
filled_data = resampled_data
# 绘制折线图
data2 = pd.read_csv('D:\Program Files\JetBrains\pythonProject\huaweiyuce/alldata.csv', header=1, encoding='latin1')
# 提取第一列的数据，并将其作为datetime列
datetime_column2 = pd.to_datetime(data2.iloc[:, 0], errors='coerce')
# 创建包含'datetime'和'value'两列的DataFrame
series_2 = pd.DataFrame({'datetime': datetime_column2, 'value': data2.iloc[:, 1]})
# 将datetime列设置为索引
series_2 = series_2.set_index('datetime')
chuangkou = 190
series = series_2.iloc[:chuangkou, :]
true_price = series.iloc[1:188,0]

# 绘制图形
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制 filled_data 的折线图，并设置标签
filled_data.plot(kind='line', ax=ax, label='Predicted Prices')
# 绘制 true_price 的折线图，并设置标签
true_price.plot(kind='line', ax=ax, label='True Prices')
# 设置图形标题和轴标签
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Filled Data')
# 显示网格
plt.grid(True)
# 显示图例
plt.legend()
# 显示图形
plt.show()

y_test = true_price
predicted_prices = filled_data.iloc[:, 0]

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
#这个analyse2是做的不差分的数据。
#是预测一个填一个数据，而analyse是单纯的直接采用前面的数据。
#那这个3，我想想能不能直接去差分吧，预测一个，差分一次，注意模型，要用差分的模型，先改模型