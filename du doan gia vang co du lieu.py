import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM


# Phần 1: Đọc dữ liệu từ file csv và tiền xử lý dữ liệu.

df = pd.read_csv('D:/file hoc tap/hoc ki 5/tri tue nhan tao/file thuc hanh/sklearn/Gold Price (2013-2023).csv')
df.drop(['Vol', 'Change %'], axis=1, inplace=True) # Xóa cột 'Vol' và 'Change %' trong file csv.
df['Date'] = pd.to_datetime(df['Date']) # Chuyển cột 'Date' trong file csv thành kiểu dữ liệu datetime.
df.sort_values(by='Date', ascending=True, inplace=True) # Sắp xếp lại cột 'Date' trong file csv theo thứ tự tăng dần. 
df.reset_index(drop=True, inplace=True) # Reset lại index của file csv sau khi đã sắp xếp lại cột 'Date'.                
NumCols = df.columns.drop(['Date']) # Lấy tất cả các cột trong file csv ngoại trừ cột 'Date'.                 
df[NumCols] = df[NumCols].replace({',': ''}, regex=True) # Xóa dấu ',' trong các cột trong file csv.
df[NumCols] = df[NumCols].astype('float64') # Chuyển các cột trong file csv thành kiểu dữ liệu float64.


# Phần 2: Vẽ biểu đồ dữ liệu với cột 'Date' làm trục x và cột 'Price' làm trục y.

fig = px.line(y=df.Price, x=df.Date) # Vẽ biểu đồ đường với trục x là cột 'Date' và trục y là cột 'Price'.
fig.update_traces(line_color='black') # Màu đen cho đường biểu đồ.

# Cập nhật tiêu đề và màu nền cho biểu đồ.
fig.update_layout(xaxis_title="Date", # Tiêu đề trục x.
                  yaxis_title="Scaled Price", # Tiêu đề trục y.
                  title={'text': "Gold Price History Data", 'y':0.95, 'x':0.5}, # Tiêu đề của biểu đồ ở vị trí (0.5, 0.95).
                  plot_bgcolor='rgba(255,223,0,0.8)') # Màu nền của biểu đồ.
fig.show()                  # Hiển thị biểu đồ


# Phần 3: Vẽ biểu đồ với 2 đường biểu diễn trước năm 2022 và sau năm 2022.

test_size = df[df.Date.dt.year==2022].shape[0] # Số lượng hàng có năm 2022 trong file csv.

# plt là thư viện vẽ biểu đồ trong Python. 
# rc là viết tắt của 'run command', có thể hiểu là 'thay đổi cấu hình'.
plt.figure(figsize=(15, 6), dpi=150) # Kích thước 15x6 inch với độ phân giải 150 dpi.
plt.rc('axes', facecolor='yellow') # Màu nền của biểu đồ.
plt.rc('axes', edgecolor='white') # Màu viền của biểu đồ.

# plt.plot() vẽ biểu đồ dữ liệu với cột 'Date' làm trục x và cột 'Price' làm trục y. 
# color là màu của đường biểu đồ. lw là độ rộng của đường biểu đồ.

# df.Date[:-test_size] là cột 'Date' trừ năm 2022, df.Price[:-test_size] là cột 'Price' trừ năm 2022. 
plt.plot(df.Date[:-test_size], df.Price[:-test_size], color='black', lw=2) 

# df.Date[-test_size:] là cột 'Date' của năm 2022, df.Price[-test_size:] là cột 'Price' của năm 2022.
plt.plot(df.Date[-test_size:], df.Price[-test_size:], color='blue', lw=2)

plt.title('Gold Price Training and Test Sets', fontsize=15) # Tiêu đề của biểu đồ với kích thước chữ 15.
plt.xlabel('Date', fontsize=12) # Tên trục x là 'Date' với kích thước chữ 12.
plt.ylabel('Price', fontsize=12) # Tên trục y là 'Price' với kích thước chữ 12.

# Chú thích cho biểu đồ với tên 'Training set' và 'Test set' ở vị trí trên bên trái với kích thước chữ 15.
plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15}) 

plt.grid(color='white') # Màu của lưới là trắng.
plt.show()              # Hiển thị biểu đồ.



# Phần 4: Tiền xử lý dữ liệu và chia dữ liệu thành tập huấn luyện và tập kiểm tra.

scaler = MinMaxScaler() # MinMaxScaler là một phương pháp chuẩn hóa dữ liệu giữa 0 và 1.
scaler.fit(df.Price.values.reshape(-1,1)) # Chuẩn hóa cột 'Price' trong file csv.

window_size = 60 # Là số lượng dữ liệu được sử dụng để dự đoán dữ liệu tiếp theo.

train_data = df.Price[:-test_size] # Tập huấn luyện là dữ liệu từ năm 2013 đến năm 2021.
train_data = scaler.transform(train_data.values.reshape(-1,1)) # Chuẩn hóa tập huấn luyện.

X_train = [] # Dữ liệu đầu vào của tập huấn luyện.
y_train = [] # Dữ liệu đầu ra của tập huấn luyện.

# Vòng lặp để tạo dữ liệu đầu vào và đầu ra của tập huấn luyện từ năm 2013 đến năm 2021.
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

test_data = df.Price[-test_size-window_size:] # Tập kiểm tra là dữ liệu từ window_size ngày trước năm 2022 đến năm 2022. 
test_data = scaler.transform(test_data.values.reshape(-1,1)) # Chuẩn hóa tập kiểm tra.

X_test = [] # Dữ liệu đầu vào của tập kiểm tra.
y_test = [] # Dữ liệu đầu ra của tập kiểm tra.

# Vòng lặp để tạo dữ liệu đầu vào và đầu ra của tập kiểm tra từ window_size ngày trước năm 2022 đến năm 2022.
for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-window_size:i, 0])
    y_test.append(test_data[i, 0])

# Chuyển dữ liệu thành mảng numpy.
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

# Reshape dữ liệu thành 3 chiều để phù hợp với mô hình LSTM.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (-1,1))
y_test  = np.reshape(y_test, (-1,1))

# In ra kích thước của dữ liệu tập huấn luyện và tập kiểm tra.
print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape:  ', X_test.shape)
print('y_test Shape:  ', y_test.shape)


# Phần 5: Xây dựng mô hình LSTM và huấn luyện mô hình.

# Hàm tính toán độ chính xác của mô hình.
def define_model():
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = 64, return_sequences=True)(input1)  
    x = Dropout(0.2)(x)
    x = LSTM(units = 64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)
    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()
    return model

model = define_model() # Gọi hàm define_model() để xây dựng mô hình LSTM.

# Huấn luyện mô hình với dữ liệu tập huấn luyện.
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
      
result = model.evaluate(X_test, y_test) # Đánh giá mô hình với dữ liệu tập kiểm tra.
y_pred = model.predict(X_test) # Dự đoán giá vàng với dữ liệu tập kiểm tra.

MAPE = mean_absolute_percentage_error(y_test, y_pred) # Tính toán độ lỗi tuyệt đối trung bình.
Accuracy = 1 - MAPE # Tính toán độ chính xác của mô hình.

print("Test Loss:", result) # In ra kết quả đánh giá mô hình.
print("Test MAPE:", MAPE) # In ra độ lỗi tuyệt đối trung bình.
print("Test Accuracy:", Accuracy) # In ra độ chính xác của mô hình.


# Phần 6: Vẽ biểu đồ với dữ liệu thực tế và dữ liệu dự đoán.

# Chuyển dữ liệu về dạng ban đầu trước khi chuẩn hóa.
y_test_true = scaler.inverse_transform(y_test) 
y_test_pred = scaler.inverse_transform(y_pred) 
train_data = scaler.inverse_transform(train_data)

plt.figure(figsize=(15, 6), dpi=150) # Kích thước 15x6 inch với độ phân giải 150 dpi.

plt.rc('axes', facecolor='yellow') # Màu nền của biểu đồ.
plt.rc('axes', edgecolor='white') # Màu viền của biểu đồ.

# plt.plot() vẽ biểu đồ dữ liệu với cột 'Date' làm trục x và cột 'Price' làm trục y.
plt.plot(df['Date'][:-test_size], train_data, color='black', lw=2) # Đường biểu đồ tập huấn luyện.
plt.plot(df['Date'][-test_size:], y_test_true, color='blue', lw=2) # Đường biểu đồ dữ liệu thực tế của tập kiểm tra.
plt.plot(df['Date'][-test_size:], y_test_pred, color='red', lw=2) # Đường biểu đồ dữ liệu dự đoán của tập kiểm tra.


plt.title('Model Performance on Gold Price Prediction', fontsize=15) # Tiêu đề của biểu đồ với kích thước chữ 15.
plt.xlabel('Date', fontsize=12) # Tên trục x là 'Date' với kích thước chữ 12.
plt.ylabel('Price', fontsize=12) # Tên trục y là 'Price' với kích thước chữ 12.

# Chú thích cho biểu đồ với tên 'Training Data', 'Actual Test Data' và 'Predicted Test Data' ở vị trí trên bên trái với kích thước chữ 15.
plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})

plt.grid(color='white') # Màu của lưới là trắng.
plt.show()              # Hiển thị biểu đồ.







