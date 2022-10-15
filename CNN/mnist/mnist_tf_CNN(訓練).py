import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
x_test = x_test.reshape(x_test.shape[0], 28,28, 1)
print(x_train.shape)  # (60000,28,28,1)
print(x_test.shape)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# 將數字轉為 One-hot 向量
y_train2 = tf.keras.utils.to_categorical(y_train, 10)
y_test2 = tf.keras.utils.to_categorical(y_test, 10)

# 建立模型
model = tf.keras.models.Sequential()

# 加入 2D 的 Convolution Layer，接著一層 ReLU 的 Activation 函數
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(28,28,1)))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=9, kernel_size=(2, 2),padding="same", activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(2, 2),padding="same", activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.softmax ))

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train2,
          batch_size=10000,
          epochs=100,
          verbose=1)

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)

print("Ans:",np.argmax(predict,axis=-1))

print("y_test",y_test[:])





#保存模型架構
with open("model.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model.h5")
