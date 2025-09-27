import base64
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _ensure_base64_padding(data: str) -> str:
    """Ensure base64 string has proper padding for decoding."""
    padding = len(data) % 4
    if padding:
        data += "=" * (4 - padding)
    return data


# PUBLIC_INTERFACE
def split_data_uri(image_base64: str) -> Tuple[Optional[str], str]:
    """Split a data URI into (mime_type, base64_data). If not a data URI, returns (None, input)."""
    if image_base64.startswith("data:"):
        try:
            header, b64 = image_base64.split(",", 1)
            mime = header.split(";")[0][5:]  # after 'data:'
            return mime, b64
        except Exception:
            # Not well-formed, treat as raw base64
            return None, image_base64
    return None, image_base64


# PUBLIC_INTERFACE
def decode_image_base64_to_pil(image_base64: str) -> Tuple[Image.Image, bytes]:
    """Decode a base64 (or data URI) string to a PIL Image and return (image, original_bytes)."""
    _, b64 = split_data_uri(image_base64)
    b64 = _ensure_base64_padding(b64.strip())
    try:
        img_bytes = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise ValueError("Invalid base64 image input") from exc
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Unable to decode image bytes") from exc
    return img, img_bytes


# PUBLIC_INTERFACE
def pil_to_rgb_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to a NumPy RGB array."""
    return np.array(image)  # PIL's convert('RGB') ensures RGB order


# PUBLIC_INTERFACE
def mask_to_png_data_uri(mask: np.ndarray) -> str:
    """Encode a (H, W) uint8 mask (values 0..255) as PNG base64 data URI."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    pil_mask = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
