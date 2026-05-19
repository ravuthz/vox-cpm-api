from pydantic import BaseModel
from typing import Optional
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
    prompt_text: str = ""
    control_instruction: str = ""
    cfg_value: float = 2.0
    inference_timesteps: int = 10


class TTSRequest(BaseModel):
    text: str
    reference_file: str = "clone-voice.mp3"
    control_instruction: str = ""
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    output_file: str = "result.wav"
