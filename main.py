import cv2
import sys
import dlib

if len(sys.argv) > 1:
    cascPath = sys.argv[1]
else:
    cascPath = "/Users/greatraksin/PycharmProjects/neural/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)  # загрузка модели
video_capture = cv2.VideoCapture(0)

while True:
    # забираем изображение по кадрам
    ret, frame = video_capture.read()  # считывание изображения с камеры
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # рисуем квадрат вокруг башки
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        age_predictor = dlib.shape_predictor('/Users/greatraksin/PycharmProjects/neural/shape_predictor_68_face_landmarks.dat')
        landmarks = age_predictor(gray, face)
        age = dlib.face_recognition_age.compute_age(landmarks)
        cv2.putText(frame, f'Возраст {age}', (x - y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # показываем изображение
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()  # выключение камеры
cv2.destroyAllWindows()  # завершение всех процессов opencv

