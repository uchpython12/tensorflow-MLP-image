import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 顯示其中的圖形
num=0
for num in range(0,36):
   plt.subplot(6,6,num+1)
   plt.title('[%d]->%d'% (num, y_train[num]))
   plt.imshow(x_train[num])
#plt.show()


x_train = x_train.reshape(x_train.shape[0], 32,32, 3)
x_test = x_test.reshape(x_test.shape[0], 32,32, 3)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# 將數字轉為 One-hot 向量
y_train2 = tf.keras.utils.to_categorical(y_train, 10)
y_test2 = tf.keras.utils.to_categorical(y_test, 10)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(32,32,3)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.softmax ))

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1/255),
              metrics=['accuracy'])
try:
    with open('model-cifar-10.h5', 'r') as load_weights:
        model.load_weights("model-cifar-10.h5")        # 讀取模型權重

except IOError:
    print("File not exists")
# 保存模型架構
with open("model-cifar-10.json", "w") as json_file:
   json_file.write(model.to_json())
# 訓練模型
"""
for step in range(2500):
    cost = model.train_on_batch(x_train, y_train2)
    print("step{}   train cost{}".format(step, cost))
    if step % 20 == 0:
        # 保存模型權重
        model.save_weights("model-cifar-10.h5")
"""
# 訓練模型
model.fit(x_train, y_train2,
          batch_size=1000,
          epochs=100,
          verbose=1)

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict,axis=-1))
print("y_test",y_test[:])

#保存模型權重
model.save_weights("model-cifar-10.h5")
