from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import get_settings
from src.api.routers.detect import router as detect_router

settings = get_settings()

openapi_tags = [
    {
        "name": "Hand Detection",
        "description": "Endpoints for detecting hand landmarks and segmentation mask.",
    }
]

app = FastAPI(
    title="WristFit Hand Detection API",
    description=(
        "Serverless backend that proxies Perfect Corp.'s hand-tracking API and falls back to "
        "MediaPipe Hands if no credential is configured. Accepts base64 images and returns "
        "landmarks, segmentation mask, and confidence."
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allow_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Hand Detection"])
def health_check():
    """Health check endpoint indicating service status and provider availability."""
    return {
        "message": "Healthy",
        "environment": settings.environment,
        "perfectcorp_configured": bool(settings.perfectcorp_api_key),
    }


# Register routers
app.include_router(detect_router, prefix="/api")
