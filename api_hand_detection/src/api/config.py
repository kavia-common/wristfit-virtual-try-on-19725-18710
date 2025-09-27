import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Configuration settings for the Hand Detection API read from environment variables."""

    # External service (Perfect Corp) configuration
    perfectcorp_api_key: Optional[str]
    perfectcorp_api_url: str

    # Behavior
    prefer_external_by_default: bool
    external_http_timeout_seconds: float

    # CORS and security
    allow_origins: str

    # Misc
    environment: str


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# PUBLIC_INTERFACE
def get_settings() -> Settings:
    """Return application settings loaded from environment variables."""
    return Settings(
        perfectcorp_api_key=os.getenv("PERFECT_CORP_API_KEY"),
        perfectcorp_api_url=os.getenv(
            "PERFECT_CORP_API_URL",
            "https://api.perfectcorp.com/v1/hand/detect",
        ),
        prefer_external_by_default=_get_env_bool("PREFER_EXTERNAL", True),
        external_http_timeout_seconds=float(os.getenv("EXTERNAL_HTTP_TIMEOUT", "12")),
        allow_origins=os.getenv("ALLOW_ORIGINS", "*"),
        environment=os.getenv("ENVIRONMENT", "development"),
    )
