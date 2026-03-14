import time

import cv2
# import os
import numpy as np
from tensorflow.keras.models import load_model

model_best = load_model(".//FER_B.keras")

class_names=['angry', 'happy', 'neutral', 'sad', 'surprise']
haar_cascade_path = ".//haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        face_image = cv2.resize(face_roi, (48, 48))

        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype("float32") / 255.0
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        predictions = model_best.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]

        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)
    time.sleep(0.1)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
