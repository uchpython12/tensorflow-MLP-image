import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context #mac需要用這行指令下載資料集

# 載入資料（將資料打散，放入 train 與 test 資料集）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print('x_train = ' + str(x_train.shape))
print('y_train = ' + str(y_train.shape))

# 顯示其中的圖形
num=2
plt.title('x_train[%d]  Label: %d' % (num, y_train[num]))
plt.imshow(x_train[num])   # 真正的灰階樣子
plt.show()

print('x_train before reshape:', x_train.shape)
# 將原始資料轉為正確的影像排列方式
img_rows=x_train.shape[1]
img_cols=x_train.shape[2]
dim=img_rows*img_cols*3                # <---- 注意 因為圖片是彩色 所以是3 RGB
x_train = x_train.reshape(x_train.shape[0], dim)
x_test = x_test.reshape(x_test.shape[0], dim)
print('x_train after reshape:', x_train.shape)

# 影像的類別數目
num_classes = 10

# 輸入的手寫影像解析度
img_rows, img_cols = 28, 28

# 標準化輸入資料
print('x_train before div 255:',x_train[0][180:195])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 #x_train= x_train / 255
x_test /= 255
print('x_train before div 255 ', x_train[0][180:195])

print('y_train shape:', y_train.shape)
print(y_train[:10])
# 將數字轉為 One-hot 向量 (熱編碼)
category=10
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)
print("y_train2 to_categorical shape=",y_train2.shape)     #輸出 (60000, 10)
print(y_train2[:10])

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu,
    input_dim=dim))  # 784=28*28
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])
# 設定模型的 Loss 函數、Optimizer 以及用來判斷模型好壞的依據（metrics）

# 顯示模型
model.summary()

# 訓練模型
history=model.fit(x_train, y_train2,    #進行訓練的因和果的資料
          batch_size=1000,                              #設定每次訓練的筆數
          epochs=200,                       #設定訓練的次數，也就是機器學習的次數
          verbose=1)

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)        # 計算測試正確率
print("score:",score)                                                                         #輸出測試正確率
predict = model.predict(x_test)      #取得每一個結果的機率
# print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))                                                                                    #取得預測答案1
print("Ans:",np.argmax(predict,axis=-1))                                                                                    #取得預測答案1

"""
predict2 = model.predict_classes(x_test[:10])      #取得預測答案2
print("predict_classes:",predict2[:10])
"""
#輸出預測答案2
print("y_test",y_test[:10])     