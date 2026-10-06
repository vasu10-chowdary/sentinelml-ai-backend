from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class ScanContext(BaseModel):
    scan_state: str
    threat_level: str
    ml_probability: float = 0.0
    features: Optional[Dict[str, float]] = None
    files_inspected: Optional[int] = 0
    processes_analyzed: Optional[int] = 0
    ml_executed: Optional[bool] = False
    error_message: Optional[str] = None
    
    # Newly added fields for comprehensive telemetry
    directories_inspected: Optional[int] = 0
    scan_duration: Optional[float] = 0.0
    skipped_files: Optional[int] = 0
    access_denied: Optional[int] = 0
    entropy_samples: Optional[int] = 0
    avg_entropy: Optional[float] = 0.0
    max_entropy: Optional[float] = 0.0
    high_entropy_region_count: Optional[int] = 0
    network_connections: Optional[int] = 0
    external_ips: Optional[int] = 0
    feature_names: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    context: Optional[ScanContext] = None
    system_instructions: Optional[str] = Field(None, max_length=2000)

class ChatResponse(BaseModel):
    success: bool
    response: str
