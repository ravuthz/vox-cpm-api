import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

MOCK_ENV_VALUES = {"1", "true", "yes", "on"}
FORCE_MOCK = os.getenv("VOXCPM_MOCK", "").lower() in MOCK_ENV_VALUES

if FORCE_MOCK:
    VoxCPM = None
    torch = None
    MODEL_AVAILABLE = False
    logger.warning("VOXCPM_MOCK is enabled. Running in MOCK mode.")
else:
    try:
        from voxcpm import VoxCPM
        import torch

        MODEL_AVAILABLE = True
    except ImportError:
        VoxCPM = None
        torch = None
        MODEL_AVAILABLE = False
        logger.warning("VoxCPM or torch not found. Running in MOCK mode.")


class TTSService:
    def __init__(self, base_dir: Path, output_dir: str = "outputs"):
        self.base_dir = base_dir.resolve()
        self.output_dir = (self.base_dir / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.model_lock = asyncio.Lock()

        self.mock_mode = FORCE_MOCK
        try:
            self.mock_delay = max(0.0, float(os.getenv("VOXCPM_MOCK_DELAY", "5")))
        except ValueError:
            logger.warning("Invalid VOXCPM_MOCK_DELAY value. Using 5 seconds.")
            self.mock_delay = 5.0

        self.device = "cpu"
        if MODEL_AVAILABLE and torch is not None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            logger.info("Device detected: %s", self.device)

    @property
    def model_available(self) -> bool:
        return MODEL_AVAILABLE

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def resolve_base_file(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def resolve_output_file(self, file_path: str) -> Path:
        path = Path("output", file_path)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def _load_model(self):
        if VoxCPM is None:
            raise RuntimeError("voxcpm is not installed")
        return VoxCPM.from_pretrained(
            "openbmb/VoxCPM2", load_denoiser=False, device=self.device
        )

    async def ensure_model_loaded(self) -> bool:
        global MODEL_AVAILABLE
        if not MODEL_AVAILABLE:
            return False
        if self.model is not None:
            return True

        async with self.model_lock:
            if self.model is not None:
                return True
            try:
                loop = asyncio.get_running_loop()
                self.model = await loop.run_in_executor(None, self._load_model)
                logger.info("VoxCPM model loaded successfully (denoiser disabled).")
                return True
            except Exception as error:
                logger.error("Failed to load VoxCPM model: %s", error)
                MODEL_AVAILABLE = False
                return False

    async def generate_to_file(
        self,
        text: str,
        output_path: str | Path,
        reference_wav_path: Optional[str | Path] = None,
        control_instruction: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("text must not be empty")
        if cfg_value <= 0:
            raise ValueError("cfg_value must be greater than 0")
        if inference_timesteps <= 0:
            raise ValueError("inference_timesteps must be greater than 0")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        reference_path = None
        if reference_wav_path is not None:
            reference_path = Path(reference_wav_path).resolve()
            if not reference_path.is_file():
                raise FileNotFoundError(f"reference audio not found: {reference_path}")

        start_time = time.perf_counter()
        formatted_text = f"({control_instruction}){text}"

        if await self.ensure_model_loaded():
            generation_kwargs: dict[str, Any] = {
                "text": formatted_text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            }
            if reference_path is not None:
                generation_kwargs["reference_wav_path"] = str(reference_path)

            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(
                None, lambda: self.model.generate(**generation_kwargs)
            )
            sample_rate = self.model.tts_model.sample_rate
            mode = "model"
        else:
            logger.info("MOCK processing TTS for text: %s...", text[:50])
            if self.mock_delay > 0:
                await asyncio.sleep(self.mock_delay)
            wav = np.random.uniform(-1, 1, 44100 * 2)
            sample_rate = 44100
            mode = "mock"

        sf.write(str(output_path), wav, sample_rate)
        process_time = time.perf_counter() - start_time

        return {
            "success": True,
            "mode": mode,
            "process_time_seconds": round(process_time, 3),
            "reference_file": str(reference_path) if reference_path else None,
            "output_file": str(output_path),
            "sample_rate": sample_rate,
        }


tts_service = TTSService(Path(__file__).resolve().parent)
