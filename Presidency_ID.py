import numpy as np
import cv2
import imutils


ID_cascade = cv2.CascadeClassifier("cascades\data\Presidency_id.xml")
camera = cv2.VideoCapture(0)

firstFrame = None
ID_exist = False

while True:

    ret, frame = camera.read()

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ID = ID_cascade.detectMultiScale(gray,
                                       1.3, 5,
                                       minSize=(120, 120))

    if len(ID) > 0:
        ID_exist = True

    for (x, y, w, h) in ID:
        frame = cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "Presidency ID"
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
    if firstFrame is None:
        firstFrame = gray
        continue


    cv2.imshow("ID Recognizer", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break



camera.release()
cv2.destroyAllWindows()