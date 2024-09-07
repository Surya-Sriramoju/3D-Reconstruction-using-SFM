from FindFeatures import *
from imageloader import *



img_load = Image_loader('imgs', 2.0)
img1 = img_load.downscale_image(cv2.imread(img_load.image_list[0]))
img2 = img_load.downscale_image(cv2.imread(img_load.image_list[1]))

features_0, features_1 = find_features(img1, img2)

pose_array = img_load.K.ravel()
transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
transform_matrix_1 = np.empty((3, 4))

pose_0 = np.matmul(img_load.K, transform_matrix_0)
pose_1 = np.empty((3, 4)) 
total_points = np.zeros((1, 3))
total_colors = np.zeros((1, 3))

essential_matrix, em_mask = cv2.findEssentialMat(features_0, features_1, img_load.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
features_0 = features_0[em_mask.ravel()==1]
features_1 = features_1[em_mask.ravel()==1]
_, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, features_0, features_1, img_load.K)
features_0 = features_0[em_mask.ravel() > 0]
features_1 = features_1[em_mask.ravel() > 0]
transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())








