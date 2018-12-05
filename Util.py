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
        
class BaseFeatureExtractor:
    def __init__(self):
        pass
    def extractFeatures(self, image):
        #Returns a list of key points and descriptors
        pass
        

class BaseHomographyGenerator:
    def __init__(self):
        pass
    def findHomography(kp1, desc1, kp2, desc2):
        #Returns a Homography
        pass
    
class SIFTFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
    def extractFeatures(self, image):
        return self.sift.detectAndCompute(image, None)
    
class BFMatcherHomographyGenerator(BaseHomographyGenerator):
    def __init__(self):
        self.matcher = cv2.BFMatcher()
        pass
    def findHomography(self, kp1, desc1, kp2, desc2):
        matches =  self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m in matches:
            if m[0].distance < 0.5 * m[1].distance:
                good.append(m)
        matches = np.asarray(good)
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H
        
