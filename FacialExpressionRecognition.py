# %%
import cv2
import numpy as np
import keras
from keras.models import model_from_json


emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('models/model.h5')

loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])


faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

cnt = 0
pred_result = None

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    rect = None

    cnt += 1

    for (x, y, w, h) in faces:
        rect = gray[y:y + h, x:x + w]
        x_input = cv2.resize(rect, (48, 48)).reshape(1, 48, 48, 1) / 255.
        
        if cnt % 3 == 0 or pred_result is None:
            pred_result = loaded_model.predict(x_input)[0]
        pred_sort = np.argsort(-pred_result)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i in range(7):
            if i == 0:
                cv2.putText(frame, '{}: {:.3}'.format(emotions[pred_sort[i]], pred_result[pred_sort[i]]),
                            (x + w, y + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (249, 44, 232), 2)
            else:
                cv2.putText(frame, '{}: {:.3}'.format(emotions[pred_sort[i]], pred_result[pred_sort[i]]),
                            (x + w, y + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (105, 255, 55), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
