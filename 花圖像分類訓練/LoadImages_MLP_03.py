import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image

print("#######1. 下載圖片##################")

path1=os.path.abspath(os.getcwd())  # 取現在的路徑
print(path1)

import pathlib

data_dir="datasets/images32x32/"
data_dir = pathlib.Path(data_dir)

print("圖片的路徑：",data_dir)

print("#######2. 下載圖片的 張數 ##################")
roses=list(data_dir.glob('*/*.jpg'))
image_count = len(roses)
print("此路徑下面的子檔案一共有多少張jpg圖片",image_count)
#roses = list(data_dir.glob('roses/*'))

print("#######2. 下載圖片的顯示  ##################")
import matplotlib.pyplot as plt
newsize = (32, 32)
t1=PIL.Image.open(str(roses[0]))         # 讀檔案
t1=t1.resize(newsize)                    # 調整圖片大小
t1=np.asarray(t1)                        # 資料轉成numpy
plt.imshow(t1)
#plt.show()

print("#######3. 圖片轉 numpy處理 ##################")

def AI_Files_LoadAllImages(IMAGEPATH):
    IMAGEPATH=str(IMAGEPATH)
    #IMAGEPATH = 'images'
    dirs = os.listdir(IMAGEPATH)
    X = []
    Y = []
    print(dirs)
    i = 0    # 分類答案
    for name in dirs:      # 每一個路徑
        # check if folder or file
        t1=IMAGEPATH + "/" + name
        if os.path.exists(t1) and  os.path.isdir(t1):
            import sys
            if sys.platform == "win32":
                file_paths = glob.glob(os.path.join(IMAGEPATH + "\\" + name, '*.*'))    # 找底下的*.* 的檔案
            elif sys.platform == "darwin":
                file_paths = glob.glob(os.path.join(IMAGEPATH + "/" + name, '*.*'))    # 找底下的*.* 的檔案
                # MAC OS X
            for path3 in file_paths:     # 處理每一張圖片
                path3=path3.replace("\\","/")
                try:
                    im_rgb = np.asarray(PIL.Image.open(str(path3)).resize(newsize))
                    X.append(im_rgb)
                    Y.append(i)
                    print("Y=",i," 讀取：",path3)
                except:
                    print("不是圖片檔案或無法開啟：",str(path3))
            i = i + 1

    X = np.asarray(X)     # list 轉 Numpy
    Y = np.asarray(Y)
    return X,Y

X,Y=AI_Files_LoadAllImages(data_dir)
print(X.shape,Y.shape)

print("#######4. numpy處理 ##################")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.05)
print('x_train = ' + str(x_train.shape))
print('y_train = ' + str(y_train.shape))



# 影像的類別數目
num_classes = np.unique(y_train).size
category=num_classes
# 輸入的手寫影像解析度
img_rows =x_train.shape[1]
img_cols =x_train.shape[2]
img_c=x_train.shape[3]




print('x_train before reshape:', x_train.shape)
# 將原始資料轉為正確的影像排列方式
dim=img_rows*img_cols*img_c
x_train = x_train.reshape(x_train.shape[0], dim)
x_test = x_test.reshape(x_test.shape[0], dim)
print('x_train after reshape:', x_train.shape)

# 標準化輸入資料
print('x_train before div 255:',x_train[0][180:195])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train before div 255 ', x_train[0][180:195])


print('y_train shape:', y_train.shape)
print(y_train[:10])
# 將數字轉為 One-hot 向量

y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)
print("y_train2 to_categorical shape=",y_train2.shape)     #輸出 (60000, 10)
print(y_train2[:10])


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu,
    input_dim=dim))  # 784=28*28
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1/255),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])
# 設定模型的 Loss 函數、Optimizer 以及用來判斷模型好壞的依據（metrics）


# 顯示模型
model.summary()


# 訓練模型
history=model.fit(x_train, y_train2,    #進行訓練的因和果的資料
          batch_size=20,                              #設定每次訓練的筆數
          epochs=100,                       #設定訓練的次數，也就是機器學習的次數
          verbose=1)

#測試
score = model.evaluate(x_test, y_test2, batch_size=128)        # 計算測試正確率
print("score:",score)                                                                         #輸出測試正確率
predict = model.predict(x_test)                                                     #取得每一個結果的機率
# print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))                                                                                    #取得預測答案1
print("Ans:",np.argmax(predict,axis=-1))                                                                                    #取得預測答案1

#輸出預測答案2
print("y_test",y_test[:10])                                                                    #實際測試的果