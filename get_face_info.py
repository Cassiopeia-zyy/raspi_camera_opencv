import cv2 as cv
import os

path_face = "E:/opencv+py/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

cap = cv.VideoCapture(1)

face_detector = cv.CascadeClassifier(path_face)

face_id = input('\n enter user id -->  ')

print("\n initializing face capture. Look the camera and wait please ...")

count = 0

while True:
    ret, img = cap.read()
    img = cv.flip(img, 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.imwrite("E:/opencv+py/my_face_data/id." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        count += 1

    cv.imshow('image', img)

    k = cv.waitKey(1)
    if k == 27:
        break
    elif count >= 100:
        break

print("\n finish entering face !")
cap.release()
cv.destroyAllWindows()
