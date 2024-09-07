import cv2
import numpy as np

def find_features(image_0, image_1):
    sift = cv2.SIFT_create()
    key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
    key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_0, desc_1, k=2)
    feature = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            feature.append(m)

    return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])