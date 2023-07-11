import cv2

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    faces = face_detector.detectMultiScale(frame, minNeighbors=10)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 5)
        cv2.putText(frame, 'Tommy', (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()