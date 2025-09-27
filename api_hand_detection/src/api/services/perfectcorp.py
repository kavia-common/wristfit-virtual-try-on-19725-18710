from typing import List, Optional, Tuple

import httpx

from src.api.schemas import Landmark


def _normalize_landmarks(raw_landmarks) -> List[Landmark]:
    """Normalize landmarks from external service to our schema."""
    landmarks: List[Landmark] = []
    if isinstance(raw_landmarks, list):
        for item in raw_landmarks:
            # Accept dicts with x,y,z or lists/tuples of length 3
            if isinstance(item, dict) and {"x", "y", "z"} <= set(item.keys()):
                landmarks.append(
                    Landmark(x=float(item["x"]), y=float(item["y"]), z=float(item["z"]))
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                landmarks.append(
                    Landmark(x=float(item[0]), y=float(item[1]), z=float(item[2]))
                )
    return landmarks


# PUBLIC_INTERFACE
async def detect_hand_via_perfectcorp(
    image_base64: str,
    api_url: str,
    api_key: str,
    timeout_seconds: float = 12.0,
) -> Tuple[List[Landmark], Optional[str], float]:
    """Call Perfect Corp hand detection API.

    Parameters:
    - image_base64: The base64-encoded image (may be a data URI).
    - api_url: The Perfect Corp endpoint.
    - api_key: Bearer API key for authorization.
    - timeout_seconds: Request timeout.

    Returns:
    - (landmarks, segmentation_mask_data_uri, confidence)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Avoid leaking key or details in intermediaries
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
    }
    payload = {
        "image_base64": image_base64,
        # Hint for features we expect (provider may ignore)
        "features": ["landmarks", "segmentation_mask", "confidence"],
    }

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(api_url, json=payload, headers=headers)
        # 4xx / 5xx => raise for fallback
        resp.raise_for_status()
        data = resp.json()

    # Extract fields with graceful defaults
    raw_landmarks = data.get("landmarks") or data.get("hand_landmarks")
    landmarks = _normalize_landmarks(raw_landmarks)
    mask = data.get("segmentation_mask") or data.get("mask")
    confidence = float(data.get("confidence", 0.0))

    return landmarks, mask, confidence
