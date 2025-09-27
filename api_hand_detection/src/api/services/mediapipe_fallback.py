from typing import List, Optional, Tuple

import anyio
import numpy as np

from src.api.schemas import Landmark
from src.api.utils.image_utils import pil_to_rgb_numpy, mask_to_png_data_uri


def _compute_hull_mask(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Create a binary mask of the convex hull from landmark points."""
    import cv2  # lazy import

    # Clamp to image bounds
    pts = points_xy.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    pts = pts.astype(np.int32)

    hull = cv2.convexHull(pts)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _run_mediapipe_inference(rgb_image: np.ndarray) -> Tuple[List[Landmark], Optional[str], float]:
    """Run MediaPipe Hands on an RGB image and return (landmarks, mask_data_uri, confidence)."""
    import mediapipe as mp  # lazy import

    mp_hands = mp.solutions.hands
    # static_image_mode=True is appropriate for single-frame processing
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        results = hands.process(rgb_image)

    if not results.multi_hand_landmarks:
        return [], None, 0.0

    # Take the most prominent hand (first)
    hand_landmarks = results.multi_hand_landmarks[0]
    height, width, _ = rgb_image.shape

    # Convert landmarks (normalized) to our schema
    lm_list: List[Landmark] = []
    for lm in hand_landmarks.landmark:
        lm_list.append(Landmark(x=float(lm.x), y=float(lm.y), z=float(lm.z)))

    # Build mask from convex hull of landmark pixel coordinates
    points_xy = np.array([[lm.x * width, lm.y * height] for lm in lm_list], dtype=np.float32)
    mask = _compute_hull_mask(points_xy, width=width, height=height)
    mask_data_uri = mask_to_png_data_uri(mask)

    # Confidence from handedness classification if available
    conf = 0.0
    if results.multi_handedness and results.multi_handedness[0].classification:
        conf = float(results.multi_handedness[0].classification[0].score)

    return lm_list, mask_data_uri, conf


# PUBLIC_INTERFACE
async def detect_hand_via_mediapipe(pil_image) -> Tuple[List[Landmark], Optional[str], float]:
    """Run MediaPipe Hands in a worker thread and return landmarks, mask data URI, and confidence."""
    rgb_array = pil_to_rgb_numpy(pil_image)
    # MediaPipe is CPU-bound; run in a thread to avoid blocking event loop
    return await anyio.to_thread.run_sync(_run_mediapipe_inference, rgb_array)
