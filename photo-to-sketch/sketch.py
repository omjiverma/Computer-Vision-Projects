import cv2 as cv
import imutils
def imgSketch(filepath):
    img = cv.imread(filepath)
    img = imutils.resize(img, width=480)
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_inverted = cv.bitwise_not(img_grey)
    img_blur = cv.GaussianBlur(img_inverted, (99,99),0)
    img_blur_inverted = cv.bitwise_not(img_blur)
    img_sketch=cv.divide(img_grey,img_blur_inverted, scale=256.0)
    return img_sketch

if __name__ == "__main__":
    sketch = imgSketch('./women-2.jpg')
    cv.imshow("sketch",sketch)
    cv.waitKey(0)