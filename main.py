import os
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from schemas import JobResponse, JobStatus
from manager import manager

app = FastAPI(title="VoxCPM TTS Queue API")


def _upload_path(job_id: str, label: str, filename: Optional[str]) -> str:
    suffix = os.path.splitext(filename or "")[1].lower() or ".wav"
    return os.path.join(manager.upload_dir, f"{job_id}_{label}{suffix}")


@app.post("/tts/ultimate-cloning", response_model=JobResponse)
async def ultimate_cloning(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    reference_wav: Optional[UploadFile] = File(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10)
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text must not be empty")
    if cfg_value <= 0:
        raise HTTPException(status_code=400, detail="cfg_value must be greater than 0")
    if inference_timesteps <= 0:
        raise HTTPException(status_code=400, detail="inference_timesteps must be greater than 0")

    prompt_content = await prompt_wav.read()
    if not prompt_content:
        raise HTTPException(status_code=400, detail="prompt_wav must not be empty")

    reference_content = None
    if reference_wav and reference_wav.filename:
        reference_content = await reference_wav.read()
        if not reference_content:
            raise HTTPException(status_code=400, detail="reference_wav must not be empty")

    # 1. Create a job ID
    job_id = await manager.create_job()
    
    # 2. Save uploaded files
    prompt_wav_path = _upload_path(job_id, "prompt", prompt_wav.filename)
    async with aiofiles.open(prompt_wav_path, 'wb') as out_file:
        await out_file.write(prompt_content)
        
    ref_wav_path = None
    if reference_content is not None:
        ref_wav_path = _upload_path(job_id, "ref", reference_wav.filename)
        async with aiofiles.open(ref_wav_path, 'wb') as out_file:
            await out_file.write(reference_content)

    # 3. Add task to background
    background_tasks.add_task(
        manager.process_tts,
        job_id=job_id,
        text=text,
        prompt_text=prompt_text,
        prompt_wav_path=prompt_wav_path,
        reference_wav_path=ref_wav_path,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted successfully"
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
        result_url=result_url
    )

@app.get("/tts/download/{job_id}")
async def download_result(job_id: str):
    job = manager.get_job_status(job_id)
    if not job or job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="Result not found or job not completed")
    result_path = job.get("result_path")
    if not result_path or not os.path.isfile(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=result_path,
        media_type="audio/wav",
        filename=f"output_{job_id}.wav"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
