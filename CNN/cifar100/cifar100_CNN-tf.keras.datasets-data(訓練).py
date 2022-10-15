import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
labelString = ['apple', 'aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl',
'boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee',
'clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish',
'forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion',
'lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid',
'otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit',
'raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail',
'snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger',
'tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',
]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 顯示其中的圖形
s=8
num=0
fig, axes = plt.subplots(s, s,  sharex=True, sharey=True)
for i in range(s):
    for j  in range(s):
        label=y_train[num][0]
        axes[i, j].imshow(x_train[num])
        axes[i, j].set_title('%d,%d,%s' % (num, label, labelString[label]))
        num=num+1
# plt.show()

"""
x_train=x_train[:1000]
y_train=y_train[:1000]
x_test=x_test[:100]
y_test=y_test[:100]
"""

x_train = x_train.reshape(x_train.shape[0], 32,32, 3)
x_test = x_test.reshape(x_test.shape[0], 32,32, 3)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# 將數字轉為 One-hot 向量
category=100
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)

# 建立模型
model = tf.keras.models.Sequential()

# 加入 2D 的 Convolution Layer，接著一層 ReLU 的 Activation 函數
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(32,32,3)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),padding="same", activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
try:
    with open('model-cifar-100.h5', 'r') as load_weights:
        model.load_weights("model-cifar-100.h5")
except IOError:
    print("File not exists")
# 保存模型架構
with open("model-cifar-100.json", "w") as json_file:
   json_file.write(model.to_json())
# 訓練模型
for step in range(8000):
    batchs=300
    for i in range(batchs+1):
        t1=x_train.shape[0]
        batch_size=t1//batchs
        x_train_Batch=x_train[i*batch_size:(i+1)*batch_size]
        y_train2_Batch=y_train2[i*batch_size:(i+1)*batch_size]
        cost = model.train_on_batch(x_train_Batch, y_train2_Batch)
        print("step{}   train cost{}".format(step, cost),x_train_Batch.shape,y_train2_Batch.shape,i*batch_size,(i+1)*batch_size)
    if step % 2 == 0:
        # 保存模型權重
        model.save_weights("model-cifar-100.h5")

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])

#保存模型權重
model.save_weights("model-cifar-100.h5")
