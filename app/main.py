import uuid
import logging
import httpx
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import ChatRequest, ChatResponse
from app.security import check_rate_limit
from app.gemini_client import generate_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinelml-ai-backend")

app = FastAPI(title="SentinelML AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request.state.req_id = req_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as exc:
        logger.error(f"Request {req_id} failed: {exc}")
        # Redact API key if it somehow ends up in the error string
        err_str = str(exc)
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY in err_str:
            err_str = err_str.replace(settings.GEMINI_API_KEY, "[REDACTED]")
            
        status_code = 500
        detail = "AI service encountered a temporary error."
        
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code in (401, 403):
                status_code = exc.response.status_code
                detail = "AI service authentication failed."
            elif exc.response.status_code == 429:
                status_code = 429
                detail = "AI service is temporarily busy. Please try again later."
            elif exc.response.status_code in (500, 502, 503, 504):
                status_code = 503
                detail = "AI service is currently unavailable."
        elif isinstance(exc, httpx.TimeoutException):
            status_code = 504
            detail = "AI service did not respond within 15 seconds."
        elif isinstance(exc, httpx.RequestError):
            status_code = 502
            detail = "Unable to reach SentinelML AI service."
            
        return JSONResponse(
            status_code=status_code,
            content={"success": False, "response": detail},
            headers={"X-Request-ID": req_id},
            media_type="application/json; charset=utf-8"
        )

import sys
import httpx as httpx_pkg

@app.get("/health")
async def health():
    return JSONResponse(
        content={"status": "ok", "service": "sentinelml-ai-backend"},
        media_type="application/json; charset=utf-8"
    )

@app.get("/api/v1/ai/diagnostics")
async def diagnostics():
    key = settings.GEMINI_API_KEY or ""
    prefix = key[:4] if len(key) >= 4 else None
    if not key:
        prefix = None
    
    return JSONResponse(
        content={
            "service": "sentinelml-ai-backend",
            "gemini_key_configured": bool(key),
            "gemini_key_length": len(key),
            "gemini_key_prefix": prefix,
            "gemini_model": settings.GEMINI_MODEL,
            "models_configured": [settings.GEMINI_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite"],
            "python_version": sys.version.split(" ")[0],
            "httpx_version": httpx_pkg.__version__,
            "status": "ok"
        },
        media_type="application/json; charset=utf-8"
    )

@app.post("/api/v1/ai/diagnostic-test")
async def diagnostic_test():
    prompt = "What is ransomware? Answer in one short sentence."
    try:
        response_text = await generate_response(prompt)
        return JSONResponse(
            content={
                "success": True,
                "model": settings.GEMINI_MODEL,
                "response_received": True,
                "response_length": len(response_text)
            },
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        status_code = None
        error_type = type(e).__name__
        
        # Pull response text if it exists
        error_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
            try:
                if e.response.text:
                    error_msg = f"HTTP {status_code} - {e.response.text}"
            except Exception:
                pass
                
        # Strict sanitization
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY in error_msg:
            error_msg = error_msg.replace(settings.GEMINI_API_KEY, "[REDACTED]")
            
        import re
        error_msg = re.sub(r'(AIzaSy\S+|AQ\.Ab\S+)', '[REDACTED]', error_msg)
        error_msg = re.sub(r'https://generativelanguage[^\s\'"]+', '[URL_REDACTED]', error_msg)
            
        return JSONResponse(
            content={
                "success": False,
                "error_type": error_type,
                "status_code": status_code,
                "error_message": error_msg
            },
            media_type="application/json; charset=utf-8"
        )

@app.post("/api/v1/ai/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)])
async def chat(request: ChatRequest):
    if request.context:
        ctx = request.context
        ctx_str = (
            f"Scan State: {ctx.scan_state}\n"
            f"Threat Level: {ctx.threat_level}\n"
            f"ML Executed: {ctx.ml_executed}\n"
        )
        if ctx.ml_executed:
            ctx_str += f"ML Probability: {ctx.ml_probability:.4f}\n"
        ctx_str += (
            f"Files Inspected: {ctx.files_inspected}\n"
            f"Processes Analyzed: {ctx.processes_analyzed}\n"
        )
        if ctx.features:
            ctx_str += "\nExtracted ML Features:\n"
            for k, v in ctx.features.items():
                ctx_str += f"  {k}: {v:.2f}\n"
        if ctx.error_message:
            ctx_str += f"\nScan Errors: {ctx.error_message}\n"
            
        system_instruction = (
            f"Context from SentinelML:\n{ctx_str}\n\n"
            "If the user asks about the scan, answer based strictly on this provided scan data. "
            "Do not fabricate scan values. Do not override or second-guess the SentinelML classification. "
            "Keep responses concise and professional."
        )
    else:
        system_instruction = (
            "No active scan data is available for this session. "
            "If the user specifically asks about their latest scan, why something was detected, or to analyze their system, truthfully state that no scan data is available. "
            "However, if the user asks a general cybersecurity question (e.g. 'What is ransomware?', 'How do I protect my PC?'), answer it normally and professionally."
        )

    prompt = (
        f"{system_instruction}\n\n"
        f"User Message: {request.message}"
    )

    response_text = await generate_response(prompt)
    
    return JSONResponse(
        content=ChatResponse(success=True, response=response_text).model_dump(),
        media_type="application/json; charset=utf-8"
    )
