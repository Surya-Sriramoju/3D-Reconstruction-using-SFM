from FindFeatures import *
from imageloader import *
from Triangulate import *
from ReporjectionError import *
from PnP import *
from BundleAdjustment import *
from CommonPoints import *
from PlyFormat import *
# import matplotlib.pyplot as plt
from tqdm import tqdm

enable_bundle_adjustment = True

img_load = Image_loader('imgs', 2.0)
img1 = img_load.downscale_image(cv2.imread(img_load.image_list[0]))
img2 = img_load.downscale_image(cv2.imread(img_load.image_list[1]))

feature_0, feature_1 = find_features(img1, img2)

pose_array = img_load.K.ravel()
transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
transform_matrix_1 = np.empty((3, 4))


pose_0 = np.matmul(img_load.K, transform_matrix_0)
pose_1 = np.empty((3, 4)) 
total_points = np.zeros((1, 3))
total_colors = np.zeros((1, 3))

essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, img_load.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
feature_0 = feature_0[em_mask.ravel()==1]
feature_1 = feature_1[em_mask.ravel()==1]
_, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, img_load.K)
feature_0 = feature_0[em_mask.ravel() > 0]
feature_1 = feature_1[em_mask.ravel() > 0]
transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

pose_1 = np.matmul(img_load.K, transform_matrix_1)
feature_0, feature_1, points_3d = triangulation(pose_0, pose_1, feature_0, feature_1)
error, points_3d = reprojection_error(points_3d, feature_1, transform_matrix_1, img_load.K, homogenity = 1)
print("REPROJECTION ERROR: ", error)
_, _, feature_1, points_3d, _ = PnP(points_3d, feature_1, img_load.K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)
total_images = len(img_load.image_list) - 2 
pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))
threshold = 0.5

for i in tqdm(range(total_images)):
    image_2 = img_load.downscale_image(cv2.imread(img_load.image_list[i + 2]))
    features_cur, features_2 = find_features(img2, image_2)
    if i!=0:
        feature_0, feature_1, points_3d = triangulation(pose_0, pose_1, feature_0, feature_1)
        feature_1 = feature_1.T
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
        points_3d = points_3d[:, 0, :]

    # print("Points3d: ",points_3d)
    # break
    cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = common_points(feature_1, features_cur, features_2)
    # print(features_cur[:10])
    # break
    # print("Common points type: ",type(cm_points_1))
    # print(cm_points_1[:10])
    cm_points_2 = features_2[cm_points_1]
    cm_points_cur = features_cur[cm_points_1]

    # cm_points_2 = features_2[cm_mask_1]
    # cm_points_cur = features_cur[cm_mask_1]

    rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = PnP(points_3d[cm_points_0], cm_points_2, img_load.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
    transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
    pose_2 = np.matmul(img_load.K, transform_matrix_1)

    error, points_3d = reprojection_error(points_3d, cm_points_2, transform_matrix_1, img_load.K, homogenity = 0)
    cm_mask_0, cm_mask_1, points_3d = triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
    error, points_3d = reprojection_error(points_3d, cm_mask_1, transform_matrix_1, img_load.K, homogenity = 1)
    print("Reprojection Error: ", error)
    pose_array = np.hstack((pose_array, pose_2.ravel()))

    if enable_bundle_adjustment:
        points_3d, cm_mask_1, transform_matrix_1 = bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, img_load.K, threshold)
        pose_2 = np.matmul(img_load.K, transform_matrix_1)
        error, points_3d = reprojection_error(points_3d, cm_mask_1, transform_matrix_1, img_load.K, homogenity = 0)
        print("Bundle Adjusted error: ",error)
        total_points = np.vstack((total_points, points_3d))
        points_left = np.array(cm_mask_1, dtype=np.int32)
        color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
        total_colors = np.vstack((total_colors, color_vector))
    else:
        total_points = np.vstack((total_points, points_3d[:, 0, :]))
        points_left = np.array(cm_mask_1, dtype=np.int32)
        color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
        total_colors = np.vstack((total_colors, color_vector))
    
    transform_matrix_0 = np.copy(transform_matrix_1)
    pose_0 = np.copy(pose_1)
    # plt.scatter(i, error)
    # plt.pause(0.05)

    image_0 = np.copy(img2)
    img2 = np.copy(image_2)
    feature_0 = np.copy(features_cur)
    feature_1 = np.copy(features_2)
    pose_1 = np.copy(pose_2)

print(total_points.shape, total_colors.shape)
to_ply(img_load.path, total_points, total_colors, img_load)
print("Completed Exiting ...")
np.savetxt(img_load.path + '\\res\\' + img_load.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')




