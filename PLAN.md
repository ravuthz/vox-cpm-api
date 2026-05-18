# FastAPI VoxCPM Queue System Plan

## Overview
Create a FastAPI application that handles Text-to-Speech (TTS) requests using VoxCPM's "Ultimate Cloning" feature. It will use an asynchronous queue system to handle long-running tasks.

## Components
1. **API Endpoints**:
   - `POST /tts/ultimate-cloning`: Submit a new synthesis job. Supports all parameters and file uploads. Returns a `job_id`.
   - `GET /tts/job/{job_id}`: Check status and get results (metadata + download link).
   - `GET /tts/download/{job_id}`: Download the generated audio file.

2. **Job Queue**:
   - Use `FastAPI.BackgroundTasks` for initial implementation.
   - Use a `dict` (or `SQLite` for persistence) to track job status: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`.

3. **VoxCPM Integration**:
   - Wrapper for `VoxCPM.generate()` using the provided parameters.
   - Handle file management for uploaded reference audio and generated output.

## Parameters for Ultimate Cloning
- `text`: Target text to synthesize.
- `prompt_text`: Transcript of the reference audio.
- `prompt_wav`: Reference audio file (Upload).
- `reference_wav`: Optional reference audio file for timbre (Upload).
- `cfg_value`: Classifier-Free Guidance (default: 2.0).
- `inference_timesteps`: Diffusion steps (default: 10).

## File Structure
- `main.py`: FastAPI app and routes.
- `schemas.py`: Pydantic models.
- `manager.py`: Job management and processing logic.
- `utils.py`: File handling and helpers.
- `requirements.txt`: Dependencies.
