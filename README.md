# Computer Vision & Sensor Perception

A collection of computer vision, object detection, pose-estimation, and distance-estimation projects developed during my Master's studies in Mechatronics.

The repository progresses from fundamental image processing to **KITTI-based camera, LiDAR, and object-perception experiments**.

## Projects

### Edge Detection

Implementations of:

- Sobel X and Y filtering
- Laplacian edge detection
- Canny edge detection

**Technologies:** Python, OpenCV, NumPy, Matplotlib

### ArUco Pose Estimation & Augmented Reality

An ArUco-based vision experiment covering:

- Marker detection
- Camera-model setup
- Pose estimation using `solvePnP`
- 3D point projection
- Perspective transformation
- Virtual image overlay

**Technologies:** Python, OpenCV, NumPy

### Monocular Object Distance Estimation

A KITTI-based experiment combining YOLO vehicle detection with camera calibration and image geometry to estimate object distance.

The implementation includes:

- YOLO-based car detection
- IoU matching with ground truth
- Camera intrinsic parameters
- Bounding-box-based distance estimation
- Comparison with ground-truth distance

**Technologies:** Python, YOLO, OpenCV, NumPy

### Ray-Based Ground-Plane Distance Estimation

A second distance-estimation approach using camera geometry.

The method uses the bottom-center of a detected vehicle bounding box, projects the corresponding image ray using the camera intrinsics, and estimates forward distance through ground-plane intersection.

**Technologies:** Python, YOLO, OpenCV, NumPy

### LiDAR–Camera Sensor Fusion

A KITTI-based perception pipeline combining camera detections with LiDAR measurements.

The implementation includes:

- LiDAR-to-camera coordinate transformation
- Projection of LiDAR points into the image
- YOLO-based vehicle segmentation
- Association of projected LiDAR points with detected objects
- LiDAR-based distance estimation
- IoU-based matching
- Distance-error evaluation
- Experimental result export

**Technologies:** Python, YOLO, OpenCV, NumPy, Pandas, LiDAR

## Dataset

`KITTI_Selected/` contains the selected KITTI data used for the distance-estimation experiments:

```text
KITTI_Selected/
├── calib/
├── images/
├── labels/
└── README.md
```

The folder contains the camera images, calibration data, and ground-truth labels required by the KITTI experiments.

## Repository Structure

```text
Computer_Vision/
│
├── KITTI_Selected/
├── aruco_pose_estimation_overlay.py
├── canny_edge_detection.py
├── sobel_laplacian_edge_detection.py
├── kitti_monocular_distance_estimation.py
├── kitti_ray_ground_distance_estimation.py
├── kitti_lidar_camera_sensor_fusion.py
├── yolov8n_detection.pt
└── README.md
```

## Technologies

**Programming:** Python  
**Computer Vision:** OpenCV, NumPy  
**Deep Learning:** YOLO  
**Perception:** LiDAR, camera geometry, sensor projection  
**Evaluation:** IoU, distance estimation, error analysis  
**Data Processing:** Pandas, Matplotlib

## Applications

- Advanced Driver Assistance Systems (ADAS)
- Autonomous vehicles
- Robotics
- Machine vision
- 3D perception
- Multi-sensor perception

## Author

**Rakesh Nuggehalli Ramesh**

M.Sc. Mechatronics  
Ravensburg-Weingarten University of Applied Sciences, Germany
