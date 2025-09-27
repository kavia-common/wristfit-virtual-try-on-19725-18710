from typing import List, Optional

from pydantic import BaseModel, Field


# PUBLIC_INTERFACE
class HandDetectRequest(BaseModel):
    """Request model for hand detection.

    The image_base64 may be a raw base64 string or a full data URI such as:
    data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
    """

    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded image. May include a data URI prefix "
            "(e.g., data:image/jpeg;base64,...) or be raw base64."
        ),
        examples=[
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
            "/9j/4AAQSkZJRgABAQ..."  # raw base64 without data URI
        ],
    )
    prefer_external: Optional[bool] = Field(
        None,
        description=(
            "If true, prefer the external Perfect Corp service when configured. "
            "If false, force the built-in MediaPipe fallback. "
            "If omitted, uses server default (env PREFER_EXTERNAL, default true)."
        ),
    )


# PUBLIC_INTERFACE
class Landmark(BaseModel):
    """A single hand landmark with normalized coordinates (0..1)."""

    x: float = Field(..., description="X coordinate normalized to [0,1].")
    y: float = Field(..., description="Y coordinate normalized to [0,1].")
    z: float = Field(
        ...,
        description=(
            "Z coordinate in meters normalized by image width, from MediaPipe/Provider."
        ),
    )


# PUBLIC_INTERFACE
class HandDetectResponse(BaseModel):
    """Response model containing detected landmarks, segmentation mask, and confidence."""

    landmarks: List[Landmark] = Field(
        default_factory=list,
        description="List of 21 hand landmarks for the most prominent hand. Empty if not detected.",
    )
    segmentation_mask: Optional[str] = Field(
        None,
        description="Base64 data URI (image/png) of a binary segmentation mask. May be null if not available.",
    )
    confidence: float = Field(
        0.0, description="Confidence score in [0,1] for the detection (0 if not detected)."
    )
    provider: str = Field(
        ...,
        description="Which implementation produced the result: 'perfectcorp' or 'mediapipe' or 'none'.",
        examples=["perfectcorp", "mediapipe", "none"],
    )
    width: Optional[int] = Field(
        None, description="Original image width in pixels, when available."
    )
    height: Optional[int] = Field(
        None, description="Original image height in pixels, when available."
    )


# PUBLIC_INTERFACE
class ErrorResponse(BaseModel):
    """Generic error response payload."""

    detail: str = Field(..., description="Human-readable error message.")
