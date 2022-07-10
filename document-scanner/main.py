#Steps
# Step 1: Detect edges.
# Step 2: Use the edges in the image to find the contour (outline) representing the piece of paper being scanned.
# Step 3: Apply a perspective transform to obtain the top-down view of the document.

from unicodedata import name
import cv2 as cv
import numpy as np
import imutils
from imutils.perspective import four_point_transform
def rectDoc_scanner(img_path):
    
    #Reading Image
    image = imutils.resize(cv.imread(img_path), width=600)

    #Edge Detection
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # BGR to GRAY Conversion
    blur_img = cv.GaussianBlur(gray_img, (3, 3), 0)  # Blurring Image
    edged_img = cv.Canny(blur_img, 75, 220)  #Detecting Enges with  Canny Edge

    #Finding Countours
    contours, _ = cv.findContours(edged_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True) # Sorting contour by area

    # Looping through Contours
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.05 * perimeter, True) # Approximating number of edges in contour
        if len(approx) == 4:
            doc_cnts = approx
            break
    print(perimeter)
    #Applyting Warp Prospective for image straightening
    warped = four_point_transform(image, doc_cnts.reshape(4, 2))
    return warped

if __name__ == "__main__":
    #Executed when main.py  run
    scanned = rectDoc_scanner('./book.jpg')
    cv.imshow("Scanned", cv.resize(scanned, (400, 600)))
    cv.waitKey(0)
    cv.destroyAllWindows()




