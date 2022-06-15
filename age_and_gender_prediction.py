import cv2 as cv
import numpy as np


age_model = cv.dnn.readNetFromCaffe(
    "age.prototxt", "dex_chalearn_iccv2015.caffemodel")
gender_model = cv.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")

default_detector_path = "haarcascade_frontalface_default.xml"

detector = cv.CascadeClassifier(default_detector_path)


def get_predection(frame):

    faces = detector.detectMultiScale(frame, 1.5, 6)
    # print(faces)
    x, y, w, h = faces[0]
    bounding_box = [x, y, w, h]
    detected_face = frame[y:y+h, x:x+w]
    detected_face = cv.resize(detected_face, (224, 224))
    detected_face_blob = cv.dnn.blobFromImage(detected_face)
    age_model.setInput(detected_face_blob)
    age_predection = age_model.forward()

    gender_model.setInput(detected_face_blob)
    gender_predection = gender_model.forward()
    
    return bounding_box, age_predection, gender_predection


indexes = np.array([i for i in range(0, 101)])

video = cv.VideoCapture(0)
while True:
    ret, frame = video.read()
    bounding_box, age_predection, gender_predection = get_predection(frame)

    apparent_age = round(np.sum(age_predection[0] * indexes))
    gender = "Female" if np.argmax(gender_predection[0]) == 0 else "Male"

    x, y, w, h = bounding_box
    label = "{},{}".format(gender, apparent_age)
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.putText(frame, label, (x, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.imshow("Age and Gender Predection", frame)
    wait_key = cv.waitKey(1)
    if wait_key == ord('q'):
        break
video.release()
cv.destroyAllWindows()
