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

### 1. Generate TTS

`POST /tts/generate`

Canonical JSON service endpoint. It follows the working VoxCPM generation path:
`text`, optional `control_instruction`, `reference_file`, `cfg_value`, and
`inference_timesteps`.

**Body (JSON):**

- `text`: The target text to synthesize.
- `reference_file`: Reference audio path relative to the project root, or an absolute path.
- `control_instruction` (Optional): Voice/style instruction.
- `cfg_value` (Optional): Classifier-Free Guidance value (default: 2.0).
- `inference_timesteps` (Optional): Number of diffusion steps (default: 10).
- `output_file` (Optional): Output WAV path (default: `result.wav`).

**Example Request (curl):**

```bash
curl -X POST "http://localhost:8000/tts/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, this is a test.",
       "reference_file": "man-aron.mp3",
       "control_instruction": "calm, natural voice",
       "output_file": "result.wav"
     }'
```

### 2. Stream TTS Runtime

`POST /tts/process`

Runs TTS immediately and streams Server-Sent Events from model loading through
completion. The final `completed` event includes a `result_url`.

**Parameters (Form Data):**

- `text`: The target text to synthesize.
- `prompt_wav`: Reference audio file (WAV).
- `prompt_text` (Optional): Legacy transcript field; accepted for older clients.
- `reference_wav` (Optional): Reference audio for timbre (if different from prompt).
- `control_instruction` (Optional): Voice/style instruction, matching `/tts/generate`.
- `cfg_value` (Optional): Classifier-Free Guidance value (default: 2.0).
- `inference_timesteps` (Optional): Number of diffusion steps (default: 10).

**Example Request (curl):**

```bash
curl -X POST "http://localhost:8000/tts/process" \
     -N \
     -H "Content-Type: multipart/form-data" \
     -F "text=Hello, this is a test of the ultimate cloning system." \
     -F "control_instruction=calm, natural voice" \
     -F "prompt_wav=@path/to/your/reference.wav"
```

Events include `job_created`, `processing`, `model_loading`, `model_ready`,
`mock_mode`, `tts_started`, `progress`, `completed`, and `failed`.

### 3. Submit Queued TTS Job

`POST /tts/ultimate-cloning`

Uses the same form data, queues the TTS job in the background, and returns a
`job_id` immediately.

### 4. Check Job Status

`GET /tts/job/{job_id}`

Returns the current status (`pending`, `processing`, `completed`, `failed`) and a `result_url` when finished.

### 5. Download Result

`GET /tts/download/{job_id}`

Downloads the generated WAV file.

## Project Structure

- `main.py`: FastAPI application and endpoint definitions.
- `service.py`: Shared TTS model loading and generation service.
- `manager.py`: Logic for job queuing, file handling, and model inference.
- `schemas.py`: Pydantic models for request/response validation.
- `PLAN.md`: Initial architecture plan.
- `uploads/`: Temporary storage for uploaded reference audio.
- `outputs/`: Storage for generated TTS audio files.
