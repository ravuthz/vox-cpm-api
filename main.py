import os
import json
import asyncio
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
from schemas import JobResponse, JobStatus, TTSRequest
from manager import manager
from service import tts_service

app = FastAPI(title="VoxCPM TTS Queue API")


def _upload_path(job_id: str, label: str, filename: Optional[str]) -> str:
    suffix = os.path.splitext(filename or "")[1].lower() or ".wav"
    return os.path.join(manager.upload_dir, f"{job_id}_{label}{suffix}")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _prepare_tts_job(
    text: str,
    prompt_text: str,
    prompt_wav: UploadFile,
    reference_wav: Optional[UploadFile],
    cfg_value: float,
    inference_timesteps: int,
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if cfg_value <= 0:
        raise HTTPException(status_code=400, detail="cfg_value must be greater than 0")
    if inference_timesteps <= 0:
        raise HTTPException(
            status_code=400, detail="inference_timesteps must be greater than 0"
        )

    prompt_content = await prompt_wav.read()
    if not prompt_content:
        raise HTTPException(status_code=400, detail="prompt_wav must not be empty")

    reference_content = None
    if reference_wav and reference_wav.filename:
        reference_content = await reference_wav.read()
        if not reference_content:
            raise HTTPException(
                status_code=400, detail="reference_wav must not be empty"
            )

    # 1. Create a job ID
    job_id = await manager.create_job()

    # 2. Save uploaded files
    prompt_wav_path = _upload_path(job_id, "prompt", prompt_wav.filename)
    async with aiofiles.open(prompt_wav_path, "wb") as out_file:
        await out_file.write(prompt_content)

    ref_wav_path = None
    if reference_content is not None:
        ref_wav_path = _upload_path(job_id, "ref", reference_wav.filename)
        async with aiofiles.open(ref_wav_path, "wb") as out_file:
            await out_file.write(reference_content)

    return job_id, prompt_wav_path, ref_wav_path


@app.post("/tts/process")
async def process_tts_stream(
    text: str = Form(...),
    prompt_text: str = Form(""),
    prompt_wav: UploadFile = File(...),
    reference_wav: Optional[UploadFile] = File(None),
    control_instruction: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    job_id, prompt_wav_path, ref_wav_path = await _prepare_tts_job(
        text=text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        reference_wav=reference_wav,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    async def event_stream():
        yield _sse("job_created", {"job_id": job_id, "status": JobStatus.PENDING})
        manager.update_job(job_id, JobStatus.PROCESSING)
        yield _sse("processing", {"job_id": job_id, "status": JobStatus.PROCESSING})

        if manager.model_available and not manager.model_loaded:
            yield _sse("model_loading", {"job_id": job_id, "device": manager.device})
            load_task = asyncio.create_task(manager.ensure_model_loaded())
            while not load_task.done():
                await asyncio.sleep(2)
                if not load_task.done():
                    yield _sse("progress", {"job_id": job_id, "stage": "model_loading"})

            if await load_task:
                yield _sse("model_ready", {"job_id": job_id, "device": manager.device})
            else:
                yield _sse(
                    "mock_mode",
                    {"job_id": job_id, "message": "VoxCPM model is unavailable"},
                )
        elif manager.model_loaded:
            yield _sse("model_ready", {"job_id": job_id, "device": manager.device})
        else:
            yield _sse(
                "mock_mode",
                {"job_id": job_id, "message": "VoxCPM model is unavailable"},
            )

        task = asyncio.create_task(
            manager.synthesize_to_file(
                job_id=job_id,
                text=text,
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
                reference_wav_path=ref_wav_path,
                control_instruction=control_instruction,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        )

        yield _sse("tts_started", {"job_id": job_id})
        while not task.done():
            await asyncio.sleep(2)
            if not task.done():
                yield _sse("progress", {"job_id": job_id, "stage": "tts_running"})

        try:
            output_path = await task
            manager.update_job(job_id, JobStatus.COMPLETED, result_path=output_path)
            yield _sse(
                "completed",
                {
                    "job_id": job_id,
                    "status": JobStatus.COMPLETED,
                    "result_url": f"/tts/download/{job_id}",
                },
            )
        except Exception as e:
            manager.update_job(job_id, JobStatus.FAILED, error=str(e))
            yield _sse(
                "failed",
                {"job_id": job_id, "status": JobStatus.FAILED, "error": str(e)},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/tts/ultimate-cloning", response_model=JobResponse)
async def submit_tts_job(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_text: str = Form(""),
    prompt_wav: UploadFile = File(...),
    reference_wav: Optional[UploadFile] = File(None),
    control_instruction: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
):
    job_id, prompt_wav_path, ref_wav_path = await _prepare_tts_job(
        text=text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        reference_wav=reference_wav,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    # 3. Add task to background
    background_tasks.add_task(
        manager.process_tts,
        job_id=job_id,
        text=text,
        prompt_text=prompt_text,
        prompt_wav_path=prompt_wav_path,
        reference_wav_path=ref_wav_path,
        control_instruction=control_instruction,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    return JobResponse(
        job_id=job_id, status=JobStatus.PENDING, message="Job submitted successfully"
    )


@app.get("/tts/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = manager.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_url = None
    if job["status"] == JobStatus.COMPLETED:
        result_url = f"/tts/download/{job_id}"

    return JobResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("error"),
        result_url=result_url,
    )


@app.get("/tts/download/{job_id}")
async def download_result(job_id: str):
    job = manager.get_job_status(job_id)
    if not job or job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=404, detail="Result not found or job not completed"
        )
    result_path = job.get("result_path")
    if not result_path or not os.path.isfile(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=result_path, media_type="audio/wav", filename=f"output_{job_id}.wav"
    )


@app.post("/tts/generate")
async def tts_run(payload: TTSRequest):
    try:
        output_path = tts_service.resolve_output_file(payload.output_file)
        reference_wav_path = tts_service.resolve_base_file(payload.reference_file)

        return await tts_service.generate_to_file(
            text=payload.text,
            output_path=output_path,
            reference_wav_path=reference_wav_path,
            control_instruction=payload.control_instruction,
            cfg_value=payload.cfg_value,
            inference_timesteps=payload.inference_timesteps,
        )

    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": str(error)},
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(error),
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
