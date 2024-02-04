"""
作者：赵江同
日期：2024年01月20日，15时：19分
"""
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score

# 读取股票价格数据
data = pd.read_csv('ec2_disk_write_bytes_1ef3de.csv')
prices = data['value'].values
time_label = data['timestamp']
# 数据预处理：归一化
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices.reshape(-1, 1))

seq_length = 10
# 划分数据集为训练集和测试集
train_size = int(len(prices) * 0.8)
train_data, test_data = prices[:train_size], prices[train_size:]
time_test_label = time_label[train_size+seq_length:]
dates = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in time_test_label]
# 创建时间窗口数据
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)


X_train = create_sequences(train_data, seq_length)
y_train = train_data[seq_length:]
X_test = create_sequences(test_data, seq_length)
y_test = test_data[seq_length:]

# 创建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=64)

# 预测未来股价
predicted_prices = model.predict(X_test)

# 反归一化预测结果
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test = scaler.inverse_transform(y_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predicted_prices)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test, predicted_prices)
print(f"R-squared (R2): {r2}")

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]

    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
    return mape

mape = calculate_mape(y_test, predicted_prices)
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")


# 绘制预测结果
plt.rc('font',family='SimHei')  #用来正常显示中文标签
plt.plot(dates,y_test, label='True Prices')
plt.plot(dates,predicted_prices,label='Predicted Prices')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('disk write  utilization')
plt.title('disk write Prediction using LSTM')
plt.show()
# Calculate absolute errors


# Plot True Prices, Predicted Prices, and Error Curve
absolute_errors = np.abs(y_test - predicted_prices)
plt.figure(figsize=(12, 6))
# Plot True Prices
plt.subplot(2, 1, 1)
plt.plot(y_test, label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('disk Price')
plt.title('disk Price Prediction using LSTM')

# Plot Error Curve
plt.subplot(2, 1, 2)
plt.plot(absolute_errors, label='Absolute Errors', color='red')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Absolute Error')
plt.title('Absolute Errors over Time')

plt.tight_layout()
plt.show()
