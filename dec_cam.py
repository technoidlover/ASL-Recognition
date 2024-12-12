#! /usr/bin/env python3

import cv2
import numpy as np
from keras.models import load_model
import time

# Constants
GESTURE_NAMES = {0: 'E', 1: 'L', 2: 'F', 3: 'V', 4: 'B'}
CAP_REGION_X_BEGIN = 0.5
CAP_REGION_Y_END = 0.8
THRESHOLD = 60
BLUR_VALUE = 41
BG_SUB_THRESHOLD = 50
LEARNING_RATE = 0
PRED_THRESHOLD = 95

# Variables
bgModel = None
isBgCaptured = 0

# Load model
model = load_model('models/mymodel.h5')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32') / 255
    pred_array = model.predict(image)
    result = GESTURE_NAMES[np.argmax(pred_array)]
    score = float(f"{max(pred_array[0]) * 100:.2f}")
    return result, score

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=LEARNING_RATE)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    return cv2.bitwise_and(frame, frame, mask=fgmask)

# Camera setup
camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(cv2.bilateralFilter(frame, 5, 50, 100), 1)
    cv2.rectangle(frame, (int(CAP_REGION_X_BEGIN * frame.shape[1]), 0),
                  (frame.shape[1], int(CAP_REGION_Y_END * frame.shape[0])), (255, 0, 0), 2)

    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(CAP_REGION_Y_END * frame.shape[0]), int(CAP_REGION_X_BEGIN * frame.shape[1]):frame.shape[1]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_VALUE, BLUR_VALUE), 0)
        _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1]) > 0.2:
            target = cv2.resize(np.stack((thresh,) * 3, axis=-1), (224, 224)).reshape(1, 224, 224, 3)
            prediction, score = predict_rgb_image_vgg(target)
            if score >= PRED_THRESHOLD:
                cv2.putText(frame, f"Sign: {prediction}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10, lineType=cv2.LINE_AA)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, BG_SUB_THRESHOLD)
        isBgCaptured = 1
        cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(2)
    elif k == ord('r'):
        bgModel = None
        isBgCaptured = 0
        cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(1)

    cv2.imshow('original', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))

cv2.destroyAllWindows()
camera.release()