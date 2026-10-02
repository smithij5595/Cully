"""
The judgment layer: asks Gemini for a structured assessment of one photo.

This is the ONLY place in the app that calls the LLM directly.
Everything else (tools.py, graph.py's check_node) works with the
structured PhotoScore this returns -- that boundary is deliberate:
LLM judgment here, plain code everywhere else.
"""
import io
import os
import random
import threading
import time

from typing import Literal, Optional

from google import genai
from google.genai import errors, types
from PIL import Image, ImageOps
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class PhotoAssessment(BaseModel):
    sharpness: int  # 1-10 -- is the SUBJECT in focus (not the background)
    framing: int  # 1-10 -- positioning, crop, horizon
    expression: int  # 1-10 -- eyes open, good moment, effort visible
    matches_preferences: bool
    reasoning: str  # one or two sentences, so a human can sanity-check it


class PhotoScore(PhotoAssessment):
    recommendation: Literal["keep", "review", "reject"]


_client: Optional[genai.Client] = None
_last_call_time = 0.0
_rate_lock = threading.Lock()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL", "0"))
MAX_IMAGE_EDGE = int(os.getenv("GEMINI_MAX_IMAGE_EDGE", "2000"))
JPEG_QUALITY = int(os.getenv("GEMINI_JPEG_QUALITY", "85"))
MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "4"))
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "MINIMAL")

def _get_client() -> genai.Client:
    """Lazily create the Gemini client so import time never needs the key."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def _prepare_image(image_path: str) -> tuple[bytes, str]:
    """Orient, resize, and compress an upload before sending it to Gemini."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return output.getvalue(), "image/jpeg"


def _wait_for_call_slot() -> None:
    """Apply optional pacing without imposing the old free-tier delay."""
    global _last_call_time
    with _rate_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()


def _is_retryable(error: Exception) -> bool:
    code = getattr(error, "code", None)
    return isinstance(error, errors.ServerError) or code in {429, 500, 502, 503, 504}


def _retry_delay(attempt: int, error: Exception) -> float:
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    return min(60.0, 2 ** (attempt - 1)) + random.uniform(0.0, 1.0)


def _parse_response(response) -> PhotoAssessment:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, PhotoAssessment):
        return parsed

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty structured response")
    return PhotoAssessment.model_validate_json(text)


PROMPT_TEMPLATE = """You are helping a photographer cull photos from a photo shoot.

Score this photo on:
If there is no human subject in the frame, give three scores of zero.
- sharpness (1-10): is the SUBJECT in sharp focus? Intentional motion blur \
in the background of a panning shot is fine and should NOT lower this \
score -- only penalize blur on the subject itself.
- framing (1-10): is the subject well-positioned, not awkwardly cropped, \
horizon straight?
- expression (1-10): if a human subject - eyes open, emotion visible.

The photographer's creative preferences for this batch: "{preferences}"

Return the three scores, whether the photo matches the preferences, and one \
or two sentences of reasoning a human could quickly sanity-check. Do not \
recommend a bucket; a deterministic tool will choose it."""


def score_photo(image_path: str, preferences: str) -> PhotoAssessment:
    """
    Sends one photo + the user's stated creative preferences to
    Gemini's vision model and returns a structured score.

    Retries transient API failures with exponential backoff and jitter.
    The pacing interval is configurable because paid-tier quotas differ.
    """
    client = _get_client()
    image_bytes, mime_type = _prepare_image(image_path)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _wait_for_call_slot()
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    PROMPT_TEMPLATE.format(
                        preferences=preferences or "no specific preferences given"
                    ),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config={
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    "thinking_config": types.ThinkingConfig(
                        thinking_level=THINKING_LEVEL
                    ),
                    "response_mime_type": "application/json",
                    "response_schema": PhotoAssessment,
                },
            )
            return _parse_response(response)
        except (errors.ServerError, errors.ClientError, ValueError) as error:
            if attempt == MAX_ATTEMPTS or (
                not _is_retryable(error) and not isinstance(error, ValueError)
            ):
                raise
            time.sleep(_retry_delay(attempt, error))
