import os
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from schemas import JobResponse, JobStatus
from manager import manager

app = FastAPI(title="VoxCPM TTS Queue API")

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
    # 1. Create a job ID
    job_id = await manager.create_job()
    
    # 2. Save uploaded files
    prompt_wav_path = os.path.join(manager.upload_dir, f"{job_id}_prompt.wav")
    async with aiofiles.open(prompt_wav_path, 'wb') as out_file:
        content = await prompt_wav.read()
        await out_file.write(content)
        
    ref_wav_path = None
    if reference_wav and reference_wav.filename:
        ref_wav_path = os.path.join(manager.upload_dir, f"{job_id}_ref.wav")
        async with aiofiles.open(ref_wav_path, 'wb') as out_file:
            content = await reference_wav.read()
            await out_file.write(content)

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
    
    return FileResponse(
        path=job["result_path"],
        media_type="audio/wav",
        filename=f"output_{job_id}.wav"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
