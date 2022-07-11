import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
#img = cv.imread('idcard-2.jpg')
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    for code in decode(img):
        data=code.data.decode('utf-8')
        pts = np.array([code.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img,[pts],True,(255,0,255),2)
        rect_pts = code.rect
        cv.putText(img,data,(rect_pts[0],rect_pts[1]),cv.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)
    cv.imshow('webcam',img)
    KEY =cv.waitKey(1)
    if KEY==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
