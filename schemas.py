from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    result_url: Optional[str] = None


class UltimateCloningParams(BaseModel):
    text: str
    prompt_text: str
    cfg_value: float = 2.0
    inference_timesteps: int = 10
