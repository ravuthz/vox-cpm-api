import os
import uuid
import logging
from typing import Dict, Optional
from schemas import JobStatus
from service import tts_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, output_dir: str = "outputs", upload_dir: str = "uploads"):
        self.jobs: Dict[str, Dict] = {}
        # Use absolute paths for reliability
        self.output_dir = os.path.abspath(output_dir)
        self.upload_dir = os.path.abspath(upload_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)

    @property
    def model_available(self) -> bool:
        return tts_service.model_available

    @property
    def model_loaded(self) -> bool:
        return tts_service.model_loaded

    @property
    def device(self) -> str:
        return tts_service.device

    async def ensure_model_loaded(self) -> bool:
        return await tts_service.ensure_model_loaded()

    async def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": JobStatus.PENDING,
            "result_path": None,
            "error": None,
        }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
    ):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            if result_path is not None:
                self.jobs[job_id]["result_path"] = result_path
            if error is not None:
                self.jobs[job_id]["error"] = error

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    async def synthesize_to_file(
        self,
        job_id: str,
        text: str,
        prompt_text: str,
        prompt_wav_path: str,
        reference_wav_path: str = None,
        control_instruction: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> str:
        output_filename = f"{job_id}.wav"
        output_path = os.path.join(self.output_dir, output_filename)
        reference_path = reference_wav_path or prompt_wav_path

        await tts_service.generate_to_file(
            text=text,
            output_path=output_path,
            reference_wav_path=reference_path,
            control_instruction=control_instruction,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return output_path

    async def process_tts(
        self,
        job_id: str,
        text: str,
        prompt_text: str,
        prompt_wav_path: str,
        reference_wav_path: str = None,
        control_instruction: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ):
        self.update_job(job_id, JobStatus.PROCESSING)

        try:
            output_path = await self.synthesize_to_file(
                job_id=job_id,
                text=text,
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
                reference_wav_path=reference_wav_path,
                control_instruction=control_instruction,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
            self.update_job(job_id, JobStatus.COMPLETED, result_path=output_path)
            logger.info(f"Job {job_id} completed successfully.")

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            self.update_job(job_id, JobStatus.FAILED, error=str(e))


# Global instance
manager = JobManager()
