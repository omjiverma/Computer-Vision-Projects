"""
First, we load our model using our prototxt and model file paths. We store the model as net
Then we load the image
Extract the dimensions and create a blob
dnn.blobFromImage takes care of pre-processing which includes setting the blob dimensions and normalization.
"""
import numpy as np
import cv2 as cv

#get model files
net = cv.dnn.readNetFromCaffe('deploy.prototxt.txt', 'weights.caffemodel')
#read Image
img = cv.imread('./image.jpg')
#width and height of image
(h, w) = img.shape[:2]

blob = cv.dnn.blobFromImage(cv.resize(img, (440, 628)), 1.0,(300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()


# for i in range(0, detections.shape[2]):
# 	confidence = detections[0, 0, i, 2]
# 	if confidence > 0.5:
# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 		(x, y, x_w,y_h) = box.astype("int")

# 		text = "{:.2f}%".format(confidence * 100)
# 		y = y - 10 if y- 10 > 10 else y + 10
# 		cv.rectangle(img, (x, y), (w, h),
# 			(0, 0, 255), 2)
# 		cv.putText(img, text, (x, y),
# 			cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# # show the output image
# cv.imshow("Output", img)
# cv.waitKey(0)