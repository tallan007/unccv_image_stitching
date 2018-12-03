import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

def loadVideoFrames(file, max_frames=-1, frame_skip=1, resize=False, gray=False, normal=True, datatype=False):
    cap = cv2.VideoCapture(file)
    start = time.time()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames >= 0 and max_frames < frames:
        frames = max_frames * frame_skip
    frame_data = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if count % frame_skip == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if datatype:
                frame = frame.astype(datatype)
            if normal:
                frame = frame / 255
            frame_data.append(frame)
        
        count = count + 1
        if count >= frames:
            end = time.time()
            cap.release()
    print("Loaded in: " + str(end - start))
    return frame_data

def showImages(ims, figsize=(8,8)):
    length = len(ims)
    fig = plt.figure(figsize=figsize)
    for i in range(length):
        fig.add_subplot(1,length,i + 1)
        plt.imshow(ims[i])
        
def showPredictions(model, X, figsize=(8,8)):
    prediction = model.predict(X)
    length = len(X)
    fig = plt.figure(figsize=figsize)
    for i in range(length):
        fig.add_subplot(1,length * 2,i + 1)
        plt.imshow(X[i])
    for i in range(length):
        fig.add_subplot(1,length * 2,length + i + 1)
        plt.imshow(prediction[i])
        
