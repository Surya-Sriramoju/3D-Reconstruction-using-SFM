import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# np.seterr(divide='ignore', invalid='ignore')
def getInliers(pt1, pt2, F):
  value = np.dot(pt1.T,np.dot(F,pt2))
  return abs(value)

def normalize(points1):
  '''
  ref: https://www.youtube.com/watch?v=zX5NeY-GTO0
  '''
  mean = np.mean(points1, axis=0)
  std = np.std(points1, axis=0)
  S = np.array([[np.sqrt(2)/std[0], 0 , 0],
                [0, np.sqrt(2)/std[1], 0],
                [0, 0, 1]])
  mean_mat = np.array([[1, 0, -mean[0]],
                      [0, 1, -mean[1]],
                      [0, 0, 1]])
  T = np.matmul(S, mean_mat)
  return points1,T

def computeF(pts1, pts2):
  n = pts1.shape[0]
  A = np.zeros((n, 9))
  for i in range(n):
    A[i][0] = pts1[i][0]*pts2[i][0]
    A[i][1] = pts1[i][0]*pts2[i][1]
    A[i][2] = pts1[i][0]
    A[i][3] = pts1[i][1]*pts2[i][0]
    A[i][4] = pts1[i][1]*pts2[i][1]
    A[i][5] = pts1[i][1]
    A[i][6] = pts2[i][0]
    A[i][7] = pts2[i][1]
    A[i][8] = 1

  U, S, V = np.linalg.svd(A)
  F = V[-1].reshape(3, 3)
  U, S, V = np.linalg.svd(F)
  S[2] = 0
  F = np.dot(U, np.dot(np.diag(S), V))
  return F

def ransac_alg(points1, points2, thresh):
  max_it = 2500
  j = 0
  best_F = np.zeros((3,3))
  num_of_inliers = 0
  while j<max_it:
    pts1 = []
    pts2 = []
    random_points = random.sample(range(0, points1.shape[0]), 8)
    for i in random_points:
      pts1.append(points1[i])
      pts2.append(points2[i])
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    F = computeF(pts1,pts2)
    values = []
    
    for i in range(points1.shape[0]):
      value = getInliers(points1[i], points2[i], F)
      #print(value)
      if value<thresh:
        values.append(value)
        #print('hi')
    if (len(values)) > num_of_inliers:
      num_of_inliers = len(values)
      best_F = F
      #print(len(values))
    j += 1
  return best_F

def transform_points(P,T):
  x = np.dot(T,P.T)
  return x.T

def ransacF(points1, points2, thresh):
  P1,T1 = normalize(points1)
  P2,T2 = normalize(points2)
  P1_trans = transform_points(P1,T1)
  P2_trans = transform_points(P2,T2)
  F = ransac_alg(P1_trans,P2_trans, thresh)
  f_mat = np.dot(np.transpose(T2), np.dot(F, T1))
  return f_mat

def get_E(F, K):
   E = np.dot(K.T, np.dot(F,K))
   U,S,V = np.linalg.svd(E)
   S = [1,1,0]
   E = np.dot(U, np.dot(np.diag(S), V))
   return E

def three_point_form(points):
  points1 = np.zeros((points.shape[0],3))
  
  for i in range(points.shape[0]):
    points1[i][0] = points[i][0]
    points1[i][1] = points[i][1]
    points1[i][2] = 1
  return points1

def get_Keypoints(img1, img2):
    img1_Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_Gray,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_Gray,None)

    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    good_matches = []
    i = 0
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
            good_matches.append(m)
            i += 1
    
    source_points = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    key_pts1 = np.zeros((source_points.shape[0],2))
    key_pts2 = np.zeros((destination_points.shape[0],2))
    for i in range(source_points.shape[0]):
      key_pts1[i] = source_points[i]
      key_pts2[i] = destination_points[i]
    
    key_pts1 = three_point_form(key_pts1)
    key_pts2 = three_point_form(key_pts2)

    return key_pts1, key_pts2