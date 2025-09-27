import logging
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from src.api.config import get_settings
from src.api.schemas import (
    ErrorResponse,
    HandDetectRequest,
    HandDetectResponse,
    Landmark,
)
from src.api.services.mediapipe_fallback import detect_hand_via_mediapipe
from src.api.services.perfectcorp import detect_hand_via_perfectcorp
from src.api.utils.image_utils import decode_image_base64_to_pil

router = APIRouter(tags=["Hand Detection"])
log = logging.getLogger("hand-detection")


def _should_use_external(prefer_external: Optional[bool], default: bool) -> bool:
    if prefer_external is None:
        return default
    return bool(prefer_external)


async def _run_perfectcorp_if_configured(
    image_base64: str, prefer_external: Optional[bool]
) -> Optional[Tuple[list[Landmark], Optional[str], float]]:
    settings = get_settings()
    if not settings.perfectcorp_api_key:
        return None
    if not _should_use_external(prefer_external, settings.prefer_external_by_default):
        return None
    try:
        landmarks, mask, confidence = await detect_hand_via_perfectcorp(
            image_base64=image_base64,
            api_url=settings.perfectcorp_api_url,
            api_key=settings.perfectcorp_api_key,
            timeout_seconds=settings.external_http_timeout_seconds,
        )
        return landmarks, mask, confidence
    except Exception as exc:
        # Log minimal info, avoid exposing secrets
        log.warning("PerfectCorp detection failed, falling back to MediaPipe: %s", exc)
        return None


# PUBLIC_INTERFACE
@router.post(
    "/hand-detect",
    response_model=HandDetectResponse,
    summary="Detect hand landmarks and segmentation mask",
    description=(
        "Accepts a base64-encoded image (optionally as a data URI) and returns hand landmarks, "
        "a segmentation mask (PNG data URI), and a confidence score. "
        "If the Perfect Corp API key is configured, the server will attempt to use it; "
        "on failure or if not configured, it falls back to on-device MediaPipe Hands."
    ),
    responses={
        http_status.HTTP_200_OK: {"model": HandDetectResponse, "description": "Detection result"},
        http_status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Invalid input"},
        http_status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "External provider error (no fallback available)",
        },
    },
)
async def hand_detect(payload: HandDetectRequest) -> HandDetectResponse:
    """Endpoint to detect hand landmarks and segmentation mask.

    Parameters:
    - payload: HandDetectRequest with base64 image and optional preference to use external provider.

    Returns:
    - HandDetectResponse: Landmarks (21 points), segmentation mask as PNG data URI (if available),
      confidence score, provider indicator, and image size when available.
    """
    # Decode input image first for width/height and potential MediaPipe fallback
    try:
        pil_image, _ = decode_image_base64_to_pil(payload.image_base64)
    except ValueError as ve:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(ve)
        ) from ve

    width, height = pil_image.size

    # Try external provider first if configured and desired
    external_result = await _run_perfectcorp_if_configured(
        image_base64=payload.image_base64, prefer_external=payload.prefer_external
    )
    if external_result:
        landmarks, mask, confidence = external_result
        provider = "perfectcorp"
        # If provider emitted nothing, go to fallback
        if landmarks:
            return HandDetectResponse(
                landmarks=landmarks,
                segmentation_mask=mask,
                confidence=confidence,
                provider=provider,
                width=width,
                height=height,
            )

    # Fallback to MediaPipe
    try:
        landmarks, mask, confidence = await detect_hand_via_mediapipe(pil_image)
        provider = "mediapipe" if landmarks else "none"
        return HandDetectResponse(
            landmarks=landmarks,
            segmentation_mask=mask,
            confidence=confidence,
            provider=provider,
            width=width,
            height=height,
        )
    except Exception as exc:
        log.exception("MediaPipe fallback failed: %s", exc)
        # If we tried external first and it failed, but no result, return 502; else 500
        settings = get_settings()
        if settings.perfectcorp_api_key and _should_use_external(
            payload.prefer_external, settings.prefer_external_by_default
        ):
            raise HTTPException(
                status_code=http_status.HTTP_502_BAD_GATEWAY,
                detail="External provider failed and fallback unavailable.",
            )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hand detection failed.",
        )
