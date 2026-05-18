# VoxCPM FastAPI Queue API

This project provides a FastAPI-based queue system for the [VoxCPM](https://github.com/OpenBMB/VoxCPM) Text-to-Speech model, specifically supporting "Ultimate Cloning".

## Features

- **Asynchronous Queue**: Submit TTS jobs and check their status later.
- **Ultimate Cloning**: Support for high-fidelity voice cloning using reference audio and transcripts.
- **File Management**: Handles reference audio uploads and manages generated output files.
- **Robust API**: Includes endpoints for job submission, status tracking, and file download.

## Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install VoxCPM:
   ```bash
   pip install voxcpm
   ```
   _Note: If VoxCPM is not installed, the API will run in MOCK mode for testing._
   You can also force mock mode with `VOXCPM_MOCK=1`; use `VOXCPM_MOCK_DELAY=0`
   for fast local smoke tests.

## Running the API

Start the server using Uvicorn:

```bash
python main.py
```

The API will be available at `http://localhost:8000`. Documentation can be found at `http://localhost:8000/docs`.

## API Endpoints

### 1. Submit Ultimate Cloning Job

`POST /tts/ultimate-cloning`

**Parameters (Form Data):**

- `text`: The target text to synthesize.
- `prompt_text`: The exact transcript of the reference audio.
- `prompt_wav`: Reference audio file (WAV).
- `reference_wav` (Optional): Reference audio for timbre (if different from prompt).
- `cfg_value` (Optional): Classifier-Free Guidance value (default: 2.0).
- `inference_timesteps` (Optional): Number of diffusion steps (default: 10).

**Example Request (curl):**

```bash
curl -X POST "http://localhost:8000/tts/ultimate-cloning" \
     -H "Content-Type: multipart/form-data" \
     -F "text=Hello, this is a test of the ultimate cloning system." \
     -F "prompt_text=This is the text in the reference audio." \
     -F "prompt_wav=@path/to/your/reference.wav"
```

### 2. Check Job Status

`GET /tts/job/{job_id}`

Returns the current status (`pending`, `processing`, `completed`, `failed`) and a `result_url` when finished.

### 3. Download Result

`GET /tts/download/{job_id}`

Downloads the generated WAV file.

## Project Structure

- `main.py`: FastAPI application and endpoint definitions.
- `manager.py`: Logic for job queuing, file handling, and model inference.
- `schemas.py`: Pydantic models for request/response validation.
- `PLAN.md`: Initial architecture plan.
- `uploads/`: Temporary storage for uploaded reference audio.
- `outputs/`: Storage for generated TTS audio files.
