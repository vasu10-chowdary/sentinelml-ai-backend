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
    logger.info("AI diagnostic request started")
    if not settings.GEMINI_API_KEY:
        logger.error("AI model failed: GEMINI_API_KEY is not configured")
        raise ValueError("GEMINI_API_KEY is not configured on the backend.")

    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    last_error = ""

    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in MODELS:
            logger.info(f"AI model attempted: {model}")
            url = f"{base_url}/{model}:generateContent?key={settings.GEMINI_API_KEY}"
            
            try:
                response = await client.post(url, headers=headers, json=payload)
                logger.info(f"AI model HTTP status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info("AI model succeeded")
                        logger.info("AI diagnostic request completed")
                        return text.strip()
                    except (KeyError, IndexError):
                        err_msg = "Malformed response from Gemini API"
                        logger.error(f"AI model failed: {err_msg}")
                        raise Exception(err_msg)
                
                if response.status_code in (429, 500, 502, 503, 504, 404):
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(f"AI model failed: {last_error}. Trying next fallback...")
                    continue
                else:
                    err_body = response.text
                    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY in err_body:
                        err_body = err_body.replace(settings.GEMINI_API_KEY, "[REDACTED]")
                    import re
                    err_body = re.sub(r'(AIzaSy\S+|AQ\.Ab\S+)', '[REDACTED]', err_body)
                    err_msg = f"HTTP {response.status_code} - {err_body}"
                    logger.error(f"AI model error: {err_msg}")
                    # Provide a safe string to HTTPStatusError to avoid URL leakage in str(e)
                    raise httpx.HTTPStatusError("Gemini API rejected request", request=response.request, response=response)

            except httpx.TimeoutException:
                last_error = "TIMEOUT"
                logger.warning(f"AI model failed: {last_error}. Trying next fallback...")
                continue
            except httpx.HTTPError as e:
                # E.g. connection error
                err_type = type(e).__name__
                logger.error(f"AI model failed: HTTP error of type {err_type}")
                raise e

    logger.error(f"AI diagnostic request completed with failure: {last_error}")
    raise Exception(f"All AI models unavailable. Last error: {last_error}")
