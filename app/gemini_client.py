import httpx
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Suppress httpx INFO logs — the Gemini API key is in the URL query param
# and httpx logs the full request URL at INFO level.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

MODELS = [settings.GEMINI_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite"]

async def generate_response(prompt: str) -> str:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured on the backend.")

    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    last_error = ""

    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in MODELS:
            url = f"{base_url}/{model}:generateContent?key={settings.GEMINI_API_KEY}"
            
            try:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        return text.strip()
                    except (KeyError, IndexError):
                        raise Exception("Malformed response from Gemini API")
                
                if response.status_code in (429, 500, 502, 503, 504, 404):
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(f"Model {model} failed: {last_error}. Trying next fallback...")
                    continue
                else:
                    raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)

            except httpx.TimeoutException:
                last_error = "TIMEOUT"
                logger.warning(f"Model {model} timed out. Trying next fallback...")
                continue
            except httpx.HTTPError as e:
                # E.g. connection error
                raise e

    raise Exception(f"All AI models unavailable. Last error: {last_error}")
