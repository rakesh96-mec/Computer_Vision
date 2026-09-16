# Computer Vision & Sensor Perception

A collection of computer vision, object detection, pose estimation, and distance-estimation projects developed during my Master's studies in Mechatronics.

The repository includes experiments ranging from fundamental image processing to LiDAR–camera perception using the KITTI dataset.

---

## Projects

### 1. Edge Detection

Basic image-processing implementations using:

- Sobel X and Y filters
- Laplacian edge detection
- Canny edge detection

These experiments demonstrate fundamental image-gradient and edge-feature extraction techniques.

**Technologies:** Python, OpenCV, NumPy, Matplotlib

---

### 2. ArUco Pose Estimation & Augmented Reality

An ArUco-marker-based computer vision experiment implementing:

- ArUco marker detection
- Camera model and intrinsic-parameter setup
- Pose estimation using `solvePnP`
- 3D point projection
- Perspective transformation
- Virtual image overlay

**Technologies:** Python, OpenCV, NumPy

---

### 3. Monocular Object Distance Estimation

A KITTI-based experiment combining YOLO object detection with camera calibration to estimate the distance of detected vehicles.

The approach includes:

- YOLO-based vehicle detection
- Intersection-over-Union (IoU) matching with ground truth
- Camera intrinsic calibration
- Distance estimation from object bounding-box geometry
- Comparison of estimated distance against ground-truth values

**Technologies:** Python, YOLO, OpenCV, NumPy

---

### 4. Ray-Based Ground-Plane Distance Estimation

A second KITTI distance-estimation approach using camera geometry.

The implementation:

1. Detects vehicles using YOLO.
2. Matches detections with ground-truth bounding boxes using IoU.
3. Uses the bottom-center pixel of the detected vehicle.
4. Projects the pixel through the inverse camera intrinsic matrix.
5. Intersects the resulting 3D ray with a ground plane.
6. Compares the estimated distance with ground-truth distance.

**Technologies:** Python, YOLO, OpenCV, NumPy, camera geometry

---

### 5. LiDAR–Camera Sensor Fusion

A more advanced KITTI-based perception pipeline combining camera detections with LiDAR measurements.

The implementation includes:

- KITTI image, LiDAR and calibration data
- LiDAR-to-camera coordinate transformation
- Projection of LiDAR points into image coordinates
- YOLO-based vehicle segmentation
- Association of projected LiDAR points with detected objects
- LiDAR-based distance estimation
- IoU-based object matching
- Ground-truth distance comparison
- Distance-error and percentage-error evaluation
- Experimental result export to Excel

**Technologies:** Python, YOLO, OpenCV, NumPy, Pandas, LiDAR

---

## Dataset

The `KITTI_Selected` directory contains a selected subset of KITTI data used for the distance-estimation experiments.

```text
KITTI_Selected/
├── calib/
├── images/
├── labels/
└── README.md
