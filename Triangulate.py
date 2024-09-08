import cv2
import numpy as np

def triangulation(point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2):
    '''
    Triangulates 3d points from 2d vectors and projection matrices
    returns projection matrix of first camera, projection matrix of second camera, point cloud 
    '''
    pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
    return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])