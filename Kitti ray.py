import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ===================== CONFIG =====================
BASE_DIR = "KITTI_Selected"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
CALIB_DIR = os.path.join(BASE_DIR, "calib")

IOU_THRESHOLD = 0.5
CAMERA_HEIGHT_METERS = 1.65  # KITTI camera mounting height
# =================================================


# ------------------ IoU computation ------------------
def compute_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


# ------------------ Load labels ------------------
def load_labels(label_file):
    """
    Label format:
    car xmin ymin xmax ymax gt_distance
    """
    boxes = []
    distances = []

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue

            if parts[0].lower() != "car":
                continue

            xmin, ymin, xmax, ymax = map(float, parts[1:5])
            gt_distance = float(parts[5])

            boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            distances.append(gt_distance)

    return boxes, distances


# ------------------ Load camera intrinsics ------------------
def load_intrinsics(calib_file):
    values = []
    with open(calib_file, "r") as f:
        for line in f:
            for v in line.strip().split():
                try:
                    values.append(float(v))
                except ValueError:
                    pass

    values = np.array(values)

    if len(values) == 9:
        return values.reshape(3, 3)
    if len(values) == 12:
        return values.reshape(3, 4)[:, :3]

    raise RuntimeError(f"Invalid calibration file: {calib_file}")


# ------------------ FULL 3D RAY + GROUND INTERSECTION ------------------
def estimate_distance_ray_ground(bbox, K, camera_height):
    """
    Full 3D ray projection with ground plane intersection
    """

    # Bottom-center pixel of bounding box
    u = (bbox[0] + bbox[2]) / 2.0
    v = bbox[3]

    pixel = np.array([u, v, 1.0])
    ray_dir = np.linalg.inv(K) @ pixel

    # Invalid ray (above horizon)
    if ray_dir[1] <= 0:
        return None

    # Intersect ray with ground plane Y = camera_height
    t = camera_height / ray_dir[1]

    X = t * ray_dir[0]
    Y = t * ray_dir[1]
    Z = t * ray_dir[2]

    return Z  # forward distance in meters


# ------------------ Match detections to GT ------------------
def match_detections(detections, gt_boxes):
    matches = []
    used_gt = set()

    for det in detections:
        best_iou = 0
        best_gt = None

        for i, gt in enumerate(gt_boxes):
            if i in used_gt:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_gt is not None and best_iou >= IOU_THRESHOLD:
            matches.append((det, gt_boxes[best_gt], best_iou))
            used_gt.add(best_gt)

    return matches


# ===================== MAIN =====================
model = YOLO("yolov8n.pt")

all_gt_dist = []
all_est_dist = []

for img_name in sorted(os.listdir(IMAGE_DIR)):
    base = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    calib_path = os.path.join(CALIB_DIR, base + ".txt")

    if not (os.path.exists(label_path) and os.path.exists(calib_path)):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    gt_boxes, gt_distances = load_labels(label_path)
    K = load_intrinsics(calib_path)

    # YOLO detections
    results = model(img, verbose=True)[0]
    detections = []

    for box in results.boxes:
        if model.names[int(box.cls[0])] != "car":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2))

    matches = match_detections(detections, gt_boxes)

    print(f"{img_name}: GT={len(gt_boxes)}, TP={len(matches)}")

    # ------------------ Visualization ------------------
    vis = img.copy()

    # Ground truth boxes (RED)
    for gt in gt_boxes:
        cv2.rectangle(vis, gt[:2], gt[2:], (0, 0, 255), 2)

    # True positives (GREEN)
    for det, gt, _ in matches:
        cv2.rectangle(vis, det[:2], det[2:], (0, 255, 0), 2)

        est_d = estimate_distance_ray_ground(det, K, CAMERA_HEIGHT_METERS)
        gt_idx = gt_boxes.index(gt)
        gt_d = gt_distances[gt_idx]

        if est_d is not None:
            all_est_dist.append(est_d)
            all_gt_dist.append(gt_d)

            cv2.putText(
                vis,
                f"GT:{gt_d:.1f}m  EST:{est_d:.1f}m",
                (det[0], det[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imshow("RED = GT | GREEN = TP", vis)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# ------------------ Distance Evaluation Plot ------------------
plt.figure(figsize=(6, 6))
plt.scatter(all_gt_dist, all_est_dist, alpha=0.7)
plt.plot(all_gt_dist, all_gt_dist, "r--", label="Ideal")
plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("Estimated Distance (m)")
plt.title("Ray Projection Distance Estimation (Cars)")
plt.legend()
plt.grid()
plt.show()
