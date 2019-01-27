import numpy as np
import cv2
import os, os.path

class FaceDetector(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def get_emotion_hash(self, path, emotion_list):
        emotion_hash = {}
        for emotion in emotion_list:
            current_emotion = path + "\\" + emotion
            for image in os.scandir(current_emotion):
                if image.is_file():
                    if emotion not in emotion_hash.keys():
                        emotion_hash[emotion] = [image.name]
                    else:
                        emotion_hash[emotion].append(image.name)
        return emotion_hash

    def detect_face(self, path_to_image):
        img = cv2.imread('image')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in face:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
            region_of_interest_gray = gray[y:y+h, x:x+w]
            region_of_interest_color = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(region_of_interest_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(region_of_interest_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # cv2.destoryAllWindows()

    def analyze_emotion(self, emotion, face_detector, emotion_hash):
        for face in emotion_hash[emotion]:
            path = "organized-dataset\\{}\\{}".format(emotion, face)
            face_detector.detect_face(path)

if __name__ == "__main__":
    emotion_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    path = "organized-dataset"

    face_detector = FaceDetector()
    emotion_hash = face_detector.get_emotion_hash(path, emotion_list)

    # Test Analyze detect face for emotion: "anger"
    face_detector.analyze_faces("anger", face_detector, emotion_hash)
