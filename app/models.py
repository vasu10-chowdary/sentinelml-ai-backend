from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ScanContext(BaseModel):
    scan_state: str
    threat_level: str
    ml_probability: float = 0.0
    features: Optional[Dict[str, float]] = None
    files_inspected: Optional[int] = 0
    processes_analyzed: Optional[int] = 0
    ml_executed: Optional[bool] = False
    error_message: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    context: Optional[ScanContext] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
