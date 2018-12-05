import numpy as np
import cv2
import torch
from superpoint import *

class VideoStablization():
    def __init__(self):
        self.superpt = SuperPointFrontend(weights_path='./superpoint_v1.pth',
                      nms_dist=4,
                      conf_thresh=0.05,
                      nn_thresh=0.7,
                      cuda=False)
        self.pTracker = PointTracker(2,0.7)
    
    def convertToPoints(self, pts1, pts2, matches):
        pointsx1 = pts1[0]
        pointsy1 = pts1[1]
        pointsx2 = pts2[0]
        pointsy2 = pts2[1]
        pts1_pair = []
        pts2_pair = []
        #print(len(pointsx1))
        #print(len(pointsx2))
        #print(np.max(matches[0]))
        #print(np.max(matches[1]))
        for i in range(matches.shape[1]):
            queryIdx=int(matches[1][i])
            trainIdx=int(matches[0][i])
            
            size_1 = len(pointsx1)-1
            size_2 = len(pointsx2)-1
            
            if queryIdx>size_1 or trainIdx>size_2:
                continue

            pt1 = (int(pointsx1[queryIdx]), int(pointsy1[queryIdx]))
            pt2 = (int(pointsx2[trainIdx]), int(pointsy2[trainIdx]))
            pts1_pair.append(pt1)
            pts2_pair.append(pt2)
        return np.array(pts1_pair), np.array(pts2_pair)
        
    
    def stabilize(self, inputpath, resize=False):
        cap = cv2.VideoCapture(inputpath)
        def read():
            ret, frame = cap.read()
            if frame.shape[0] < frame.shape[1]:
                frame = cv2.transpose(frame)
                frame = cv2.flip(frame, 1)
            if resize:
                frame = cv2.resize(frame, resize)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            color = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return color, gray
        isFirst = True
        prev_pts = None
        #prev_H = np.identity(3)
        prev_H = None
        prev_desc = None
        correctedFrames = []
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        for i in range(frameCount):
            ret, frame = cap.read()
            #print("after read")
            if frame is not None:
                #print("in if loop")
                fmgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fmnorm = fmgray.astype('float32')
                pts1, desc1, heatmap1 = self.superpt.run(fmnorm/255.0)
                if isFirst:
                    prev_desc = desc1
                    prev_pts = pts1
                    isFirst = False
                    correctedFrames.append(frame)
                    desc1 = None
                    pts1 = None
                    continue
                matchesSP = self.pTracker.nn_match_two_way(prev_desc, desc1, 0.4)
                pts_pair, prev_pts_pair = self.convertToPoints(pts1, prev_pts, matchesSP)
                HG, mask = cv2.findHomography(pts_pair, prev_pts_pair, cv2.RANSAC, 5.0)
                if prev_H is None:
                    prev_H = HG
                else:
                    prev_H =  prev_H @ HG 
                prev_pts = pts1
                prev_desc = desc1
                im_corrected = cv2.warpPerspective(frame, prev_H, (frame.shape[1], frame.shape[0]))
                correctedFrames.append(im_corrected)
                count = count+1
                desc1 = None
                pts1 = None
                #print(count)
        for i in range(len(correctedFrames)):
            out.write(correctedFrames[i])
        return correctedFrames
        