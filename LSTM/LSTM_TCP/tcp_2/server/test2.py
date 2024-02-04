"""
作者：赵江同
日期：2024年02月02日，13时：44分
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 加载数据
data = pd.read_csv('D:\Program Files\JetBrains\pythonProject\huaweiyuce/alldata.csv', header=0, encoding='latin1')
datetime_column = pd.to_datetime(data.iloc[:, 0], errors='coerce')
series_1 = pd.DataFrame({'Timestamp': datetime_column, 'Value': data.iloc[:, 1]})
series = series_1.set_index('Timestamp')
series[series == 0] = pd.NA

# 使用前向填充非空值（ffill）方法填充 NaN 值
data = series.ffill()

# 随机删除一些数据以模拟不连续的时间序列
data.loc[np.random.choice(data.index, replace=False, size=20), 'Value'] = np.nan

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Value'].values.reshape(-1, 1))

# 准备数据，这里我们使用前一个时间点的数据来预测下一个时间点的数据
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 仅使用非缺失值来训练模型
mask = ~np.isnan(X.squeeze())
X_train, y_train = X[mask], y[mask]

model.fit(X_train, y_train, epochs=2, batch_size=1, verbose=2)

# 使用模型进行预测，只预测缺失值的位置
missing_indexes = data['Value'].isnull()

if len(missing_indexes) < len(X):
    missing_indexes = np.append(missing_indexes, False)

# 使用模型进行预测，只预测缺失值的位置
X_predict = X[missing_indexes]
predictions = model.predict(X_predict)
predictions = scaler.inverse_transform(predictions)

# 填充缺失值
data['Value_Predicted'] = data['Value']
data.loc[missing_indexes, 'Value_Predicted'] = predictions.squeeze()

# 输出恢复的数据
print(data[['Value', 'Value_Predicted']])