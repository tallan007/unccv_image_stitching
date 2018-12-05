

class SUPERFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.superpt = SuperPointFrontend(weights_path='./superpoint_v1.pth',
                      nms_dist=4,
                      conf_thresh=0.05,
                      nn_thresh=0.7,
                      cuda=False)
    
    def extractFeatures(self, image):
        return self.superpt.run(image)
def match(desc1, desc2):
    pTracker = PointTracker(2,0.7)
    return pTracker.nn_match_two_way(desc1, desc2, 0.4)
def convertToPoints(pts1, pts2, matches):
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