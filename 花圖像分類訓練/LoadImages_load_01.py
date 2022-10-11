import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("#######1. 下載圖片##################")

path1=os.path.abspath(os.getcwd())
print(path1)

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname="flower_photos",
                                   extract=True,
                                   cache_dir=path1,
                                   untar=True)
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
plt.show()

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