import cv2


def PnP(obj_point, image_point , K, dist_coeff, rot_vector, initial):
    '''
    Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
    returns rotational matrix, translational matrix, image points, object points, rotational vector
    '''
    if initial == 1:
        obj_point = obj_point[:, 0 ,:]
        image_point = image_point.T
        rot_vector = rot_vector.T 
    _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
    # Converts a rotation matrix to a rotation vector or vice versa
    rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

    if inlier is not None:
        image_point = image_point[inlier[:, 0]]
        obj_point = obj_point[inlier[:, 0]]
        rot_vector = rot_vector[inlier[:, 0]]
    return rot_matrix, tran_vector, image_point, obj_point, rot_vector