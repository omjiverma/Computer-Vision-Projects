import cv2 as cv
import cvzone as cvz
import time
import mediapipe as mp

cap = cv.VideoCapture('./video-2.mp4')
pTime = 0

mp_draw = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh
facemesh = mp_facemesh.FaceMesh(max_num_faces=2)
draw_specs = mp_draw.DrawingSpec((52, 235, 52),thickness=1,circle_radius=1)
while True:
    success, img = cap.read()
    img = cv.resize(img,(640,380))
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = facemesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img,face_landmark,mp_facemesh.FACEMESH_CONTOURS,draw_specs,draw_specs)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,f'FPS: {int(fps)}',(20,70),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    
    cv.imshow('Video',img)
    KEY =cv.waitKey(1)
    if KEY==ord('q'):
        break
cap.release()
cv.destroyAllWindows()