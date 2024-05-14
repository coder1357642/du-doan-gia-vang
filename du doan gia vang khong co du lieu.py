import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM

# Phần 1: Đọc dữ liệu từ file csv và tiền xử lý dữ liệu.

df = pd.read_csv('D:/file hoc tap/hoc ki 5/tri tue nhan tao/file thuc hanh/sklearn/Gold Price (2013-2023).csv')
df.drop(['Vol', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')

# Phần 2: Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
test_size = df[df.Date.dt.year==2022].shape[0]

scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1,1))

window_size = 300

train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1,1))

X_train = []
y_train = []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

test_data = df.Price[-test_size-window_size:]
test_data = scaler.transform(test_data.values.reshape(-1,1))

X_test = []
y_test = []

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-window_size:i, 0])
    y_test.append(test_data[i, 0])

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (-1,1))
y_test  = np.reshape(y_test, (-1,1))

print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape:  ', X_test.shape)
print('y_test Shape:  ', y_test.shape)

# Phần 3: Xây dựng mô hình LSTM và huấn luyện mô hình.

def define_model():
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)
    x = Dropout(0.2)(x)
    dnn_output = Dense(1)(x)
    model = Model(inputs=input1, outputs=dnn_output)
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()
    return model

model = define_model()

history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

predicted_prices = []
start_arr = X_test[0]

for i in range(test_size):
    predicted_price = model.predict(start_arr[np.newaxis, :, :])
    predicted_prices.append(predicted_price[0, 0])
    start_arr = np.append(start_arr[1:], predicted_price[0, 0]).reshape(window_size, 1)

y_pred = np.array(predicted_prices).reshape(-1, 1)

MAPE = mean_absolute_percentage_error(y_test, y_pred)
Accuracy = 1 - MAPE

print("Test MAPE:", MAPE)
print("Test Accuracy:", Accuracy)

# Phần 4: Vẽ biểu đồ với dữ liệu thực tế và dữ liệu dự đoán.

y_test_true = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)
train_data = scaler.inverse_transform(train_data)

plt.figure(figsize=(15, 6), dpi=150)

plt.plot(df['Date'][:-test_size], train_data, color='black', lw=2)
plt.plot(df['Date'][-test_size:], y_test_true, color='blue', lw=2)
plt.plot(df['Date'][-test_size:], y_pred, color='red', lw=2)

plt.title('Model Performance on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()
