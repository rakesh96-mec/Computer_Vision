import cv2
import numpy as np
from PIL.ImImagePlugin import SCALE

# ---------- CONFIG ----------
ARUCO_DICT = cv2.aruco.DICT_6X6_1000
MARKER_ID_TO_USE = None  # Set to specific ID (e.g., 0) or None for any
# ----------------------------

# Load images
input_image = cv2.imread("6.jpg")
overlay_image = cv2.imread("overlay.jpg")

if input_image is None or overlay_image is None:
    raise IOError("Could not load input or overlay image")

# Convert input to grayscale
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Load ArUco dictionary & detector
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("No ArUco markers detected")
    exit()

# Process each detected marker
for i, marker_id in enumerate(ids.flatten()):
    if MARKER_ID_TO_USE is not None and marker_id != MARKER_ID_TO_USE:
        continue

    marker_corners = corners[i][0].astype(np.float32)

    # Compute marker center
    center = marker_corners.mean(axis=0)

    # Scale corners around center
    SCALE = 5  # >1.0 extends outside marker

    marker_corners = corners[i][0].astype(np.float32)

    # Marker center
    center = marker_corners.mean(axis=0)

    # EXPAND corners outward
    expanded_marker_corners = center + SCALE * (marker_corners - center)

    # Overlay image corners
    h, w = overlay_image.shape[:2]
    overlay_corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # Homography
    H, _ = cv2.findHomography(overlay_corners, expanded_marker_corners)

    # Warp overlay
    warped_overlay = cv2.warpPerspective(
        overlay_image,
        H,
        (input_image.shape[1], input_image.shape[0])
    )

    # Create mask
    mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, expanded_marker_corners.astype(int), 255)

    mask_inv = cv2.bitwise_not(mask)

    # Remove marker area from input image
    background = cv2.bitwise_and(input_image, input_image, mask=mask_inv)

    # Add warped overlay
    foreground = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

    input_image = cv2.add(background, foreground)

# Show result
cv2.imwrite("output_overlay.png", input_image)
print("Saved output_overlay.png")