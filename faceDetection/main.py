import cv2 as cv
import mediapipe as mp
mp_face_detection  = mp.solutions.face_detection
mp_drawing= mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv.VideoCapture('./video-1.mp4')
faces=0
while True:
    success, img = cap.read()
    img = cv.resize(img,(640,380))
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if results.detections:
        faces=len(results.detections)
        for detection in results.detections:
            mp_drawing.draw_detection(img, detection,)
    
    cv.putText(img,f'Faces: {faces}',(20,70),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv.imshow('Face Detection', img)
    KEY =cv.waitKey(1)
    if KEY==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
