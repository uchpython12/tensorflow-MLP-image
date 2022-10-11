import os
import numpy as np
import PIL
import glob

def AI_Files_LoadAllImages_Conver_SizeSave(IMAGEPATH,IMAGEPATHSAVE,w,h):
    IMAGEPATH=str(IMAGEPATH)
    IMAGEPATHSAVE=str(IMAGEPATHSAVE)
    #IMAGEPATH = 'images'
    dirs = os.listdir(IMAGEPATH)
    if os.path.exists(IMAGEPATHSAVE) == False:
        os.mkdir(IMAGEPATHSAVE)
    X = []
    Y = []
    print(dirs)
    i = 0
    for name in dirs:
        # check if folder or file
        t1=IMAGEPATH + "/" + name
        t1Save=IMAGEPATHSAVE + "/" + name
        # if folder not exists them create it
        if os.path.exists(t1) and  os.path.isdir(t1):
            if os.path.exists(t1Save) == False:
                os.mkdir(t1Save)  # make a new folder
            file_paths = glob.glob(os.path.join(IMAGEPATH + "//" + name, '*.*'))
            for path3 in file_paths:
                path3=path3.replace("\\","/")
                image=PIL.Image.open(str(path3)).resize((w,h))
                im_rgb = np.asarray(image)
                X.append(im_rgb)
                Y.append(i)
                print("Y=",i," 讀取：",path3)
                saveimageFile=t1Save+ "/" +os.path.basename(path3)
                saveimageFile=saveimageFile.replace("\\","/")
                image.save(saveimageFile)

            i = i + 1

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y