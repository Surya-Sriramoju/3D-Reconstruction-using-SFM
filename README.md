# 3D Reconstruction and Bundle Adjustment

This project performs 3D reconstruction from multiple images using Structure-from-Motion (SfM) techniques, including feature extraction, triangulation, reprojection error calculation, and bundle adjustment.

## Features

- **Feature Extraction**: Detect and match keypoints between images.
- **Triangulation**: Compute 3D points from matched 2D points in stereo images.
- **Reprojection Error Calculation**: Measure how well the 3D points project back onto the image plane.
- **Pose Estimation**: Estimate the camera pose using the Perspective-n-Point (PnP) algorithm.
- **Bundle Adjustment**: Optimize the 3D reconstruction by minimizing the reprojection error across all images.
- **PLY File Export**: Save the 3D point cloud to a `.ply` file for visualization.

## File Descriptions

- **FindFeatures.py**: Contains functions to detect and match features between images.
- **imageloader.py**: Handles loading and downscaling images.
- **Triangulate.py**: Implements triangulation to compute 3D points from matched 2D points.
- **ReprojectionError.py**: Functions to calculate the reprojection error of the 3D points.
- **PnP.py**: Implements the Perspective-n-Point (PnP) algorithm for pose estimation.
- **BundleAdjustment.py**: Performs bundle adjustment to refine the 3D reconstruction.
- **CommonPoints.py**: Finds common points between images.
- **PlyFormat.py**: Converts 3D points and colors into a `.ply` file format for visualization.

## Usage

1. **Prepare Images**: Place your images in a directory named `imgs`.

2. **Run the Script**:

   Execute the script to perform 3D reconstruction and bundle adjustment:

   ```bash
   python main.py
    ```