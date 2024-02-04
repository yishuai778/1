"""
作者：赵江同
日期：2023年11月11日，11时：54分
"""
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# 读取股票价格数据
data = pd.read_csv('stock_data.csv')
prices = data['Close'].values

# 数据预处理：归一化
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices.reshape(-1, 1))

# 划分数据集为训练集和测试集
train_size = int(len(prices) * 0.8)
train_data, test_data = prices[:train_size], prices[train_size:]

# 创建时间窗口数据
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 10
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
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = calculate_mape(y_test, predicted_prices)
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
# 绘制预测结果
plt.rc('font',family='SimHei')  #用来正常显示中文标签
plt.plot(y_test, label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.show()
# Plot True Prices, Predicted Prices, and Error Curve
absolute_errors = np.abs(y_test - predicted_prices)
plt.figure(figsize=(12, 6))
# Plot True Prices
plt.subplot(2, 1, 1)
plt.plot(y_test, label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using LSTM')

# Plot Error Curve
plt.subplot(2, 1, 2)
plt.plot(absolute_errors, label='Absolute Errors', color='red')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Absolute Error')
plt.title('Absolute Errors over Time')

plt.tight_layout()
plt.show()
