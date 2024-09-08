# 3D Reconstruction and Bundle Adjustment

This project performs 3D reconstruction from multiple images using Structure-from-Motion (SfM) techniques, including feature extraction, triangulation, reprojection error calculation, and bundle adjustment.

## Features

- **Feature Extraction**: Detect and match keypoints between images.
- **Triangulation**: Compute 3D points from matched 2D points in stereo images.
- **Reprojection Error Calculation**: Measure how well the 3D points project back onto the image plane.
- **Pose Estimation**: Estimate the camera pose using the Perspective-n-Point (PnP) algorithm.
- **Bundle Adjustment**: Optimize the 3D reconstruction by minimizing the reprojection error across all images.
- **PLY File Export**: Save the 3D point cloud to a `.ply` file for visualization.

## Dependencies

- numpy
- opencv-python
- tqdm

You can install the required packages using pip:

```bash
pip install numpy opencv-contrib-python tqdm



