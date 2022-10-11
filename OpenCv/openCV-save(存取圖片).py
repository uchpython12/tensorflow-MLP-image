import cv2
from time import gmtime, strftime
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    if ret==True:
        cv2.imshow('frame',img)
    key=cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        if ret == True:
            filename1=strftime("%Y%m%d%H%M%S", gmtime())+'.jpg'
            print(filename1)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename=filename1,img=img)
cap.release()
cv2.destroyAllWindows()