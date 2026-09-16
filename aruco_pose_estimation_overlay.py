import cv2
import numpy as np

# ------------------ LOAD IMAGES ------------------
frame = cv2.imread("8.jpg")
overlay = cv2.imread("overlay.jpg")

if frame is None or overlay is None:
    raise FileNotFoundError("Image not found")

h, w = frame.shape[:2]
h_ol, w_ol = overlay.shape[:2]

# ------------------ APPROX CAMERA INTRINSICS ------------------
focal = w
camera_matrix = np.array([
    [focal, 0, w / 2],
    [0, focal, h / 2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# ------------------ ARUCO DETECTION ------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

corners, ids, _ = detector.detectMarkers(frame)

if ids is None:
    print("No marker detected")
    exit()

# ------------------ MARKER MODEL (3D) ------------------
MARKER_SIZE = 1.0  # arbitrary units

marker_3d = np.array([
    [-0.5,  0.5, 0],
    [ 0.5,  0.5, 0],
    [ 0.5, -0.5, 0],
    [-0.5, -0.5, 0]
], dtype=np.float32) * MARKER_SIZE

# Overlay plane (scaled)
scale = 3.0
plane_3d = marker_3d * scale

# Overlay source points
pts_src = np.array([
    [0, 0],
    [w_ol, 0],
    [w_ol, h_ol],
    [0, h_ol]
], dtype=np.float32)

# ------------------ POSE + PROJECTION ------------------
for i in range(len(ids)):
    pts_2d = corners[i].reshape(4, 2).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        marker_3d,
        pts_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        continue

    img_pts, _ = cv2.projectPoints(
        plane_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    pts_dst = img_pts.reshape(4, 2).astype(np.float32)

    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(overlay, H, (w, h))

    # Dynamic mask
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
    frame = cv2.add(frame, cv2.bitwise_and(warped, warped, mask=mask))

# ------------------ DISPLAY ------------------
def resize_for_display(img, max_w=1600, max_h=1080):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * s), int(h * s)))

cv2.imshow("Pose-Based Overlay (solvePnP)", resize_for_display(frame))
cv2.waitKey(0)
cv2.destroyAllWindows()
