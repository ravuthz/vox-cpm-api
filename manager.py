import os
import uuid
import asyncio
import logging
from typing import Dict, Optional
from schemas import JobStatus, JobResponse
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import VoxCPM, use a mock if not available
try:
    from voxcpm import VoxCPM
    import torch

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logger.warning("VoxCPM or torch not found. Running in MOCK mode.")


class JobManager:
    def __init__(self, output_dir: str = "outputs", upload_dir: str = "uploads"):
        global MODEL_AVAILABLE
        self.jobs: Dict[str, Dict] = {}
        # Use absolute paths for reliability
        self.output_dir = os.path.abspath(output_dir)
        self.upload_dir = os.path.abspath(upload_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)

        self.device = "cpu"
        if MODEL_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            logger.info(f"Device detected: {self.device}")

        self.model = None
        if MODEL_AVAILABLE:
            try:
                # Disable denoiser due to environment dependency issues (torchvision::nms)
                self.model = VoxCPM.from_pretrained(
                    "openbmb/VoxCPM2", load_denoiser=False, devices=self.device
                )
                logger.info("VoxCPM model loaded successfully (denoiser disabled).")
            except Exception as e:
                logger.error(f"Failed to load VoxCPM model: {e}")
                MODEL_AVAILABLE = False

    async def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": JobStatus.PENDING,
            "result_path": None,
            "error": None,
        }
        return job_id

    def update_job(
        self, job_id: str, status: JobStatus, result_path: str = None, error: str = None
    ):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            if result_path:
                self.jobs[job_id]["result_path"] = result_path
            if error:
                self.jobs[job_id]["error"] = error

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    async def process_tts(
        self,
        job_id: str,
        text: str,
        prompt_text: str,
        prompt_wav_path: str,
        reference_wav_path: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ):
        self.update_job(job_id, JobStatus.PROCESSING)

        try:
            output_filename = f"{job_id}.wav"
            output_path = os.path.join(self.output_dir, output_filename)

            if MODEL_AVAILABLE and self.model:
                # Real inference
                # Note: VoxCPM.generate is likely CPU/GPU intensive, should ideally run in a threadpool
                loop = asyncio.get_event_loop()
                wav = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        reference_wav_path=reference_wav_path or prompt_wav_path,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                    ),
                )
                sf.write(output_path, wav, self.model.tts_model.sample_rate)
            else:
                # Mock inference
                logger.info(f"MOCK processing job {job_id} for text: {text[:50]}...")
                await asyncio.sleep(5)  # Simulate processing time
                # Create a dummy wav file
                dummy_wav = np.random.uniform(-1, 1, 44100 * 2)  # 2 seconds of noise
                sf.write(output_path, dummy_wav, 44100)

            self.update_job(job_id, JobStatus.COMPLETED, result_path=output_path)
            logger.info(f"Job {job_id} completed successfully.")

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            self.update_job(job_id, JobStatus.FAILED, error=str(e))


# Global instance
manager = JobManager()
