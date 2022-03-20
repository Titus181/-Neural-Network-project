import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)  #設定GPU顯存用量按需使用
    tf.config.set_visible_devices([gpus[0]],"GPU")


import pandas            as pd
import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用來正常顯示中文標簽
plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號

from numpy                 import array
from sklearn               import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models          import Sequential
from keras.layers          import Dense,LSTM,Bidirectional


# 確保結果盡可能重現
from numpy.random          import seed
seed(1)
tf.random.set_seed(1)

# 設定相關引數
n_timestamp  = 40    # 時間
n_epochs     = 20    # 訓練數
# ====================================
#      選擇模型：
#            1: 單層 LSTM
#            2: 多層 LSTM
#            3: 雙向 LSTM
# ====================================
model_type = 1


data = pd.read_csv('1795_history_v1.csv')  # 讀取股票檔案
print(data.head())

"""
前(2453-300=2153)天的開盤價作為訓練集,后300天的開盤價作為測驗集
"""
training_set = data.iloc[300:2452, 4:5].values
test_set     = data.iloc[0:300, 4:5].values

#將資料歸一化，範圍是0到1
sc  = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled  = sc.transform(test_set)


# 取前 n_timestamp 天的資料為 X；n_timestamp+1天資料為 Y，
def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 建構 LSTM模型
if model_type == 1:
    # 單層 LSTM
    model = Sequential()
    model.add(LSTM(units=50, activation='relu',
                   input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
if model_type == 2:
    # 多層 LSTM
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(1))
if model_type == 3:
    # 雙向 LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'),
                            input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

model.summary()  # 輸出模型結構

# 該應用只觀測loss數值，不觀測準確率，所以刪去metrics選項，一會在每個epoch迭代顯示時只顯示loss值
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 損失函式用均方誤差

history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),
                    validation_freq=1)                  #測驗的epoch間隔數

model.summary()


plt.plot(history.history['loss']    , label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

predicted_stock_price = model.predict(X_test)                        # 測驗集輸入模型進行預測
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # 對預測資料還原---從（0，1）反歸一化到原始範圍
real_stock_price      = sc.inverse_transform(y_test)# 對真實資料還原---從（0，1）反歸一化到原始範圍

# 畫出真實資料和預測資料的對比曲線
plt.plot(real_stock_price, color='red', label='Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()




used_to_predict_five_days_data = data.iloc[0:100, 4:5].values
used_to_predict_five_days_data_scaled = sc.fit_transform(used_to_predict_five_days_data)


predicted_data = model.predict(used_to_predict_five_days_data_scaled)
predicted_data = sc.inverse_transform(predicted_data)
plt.plot(predicted_data, color='blue', label='Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
