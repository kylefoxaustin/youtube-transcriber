#!/usr/bin/env python3
"""
YouTube & Local Media Transcriber Web Application
FastAPI backend with real-time progress updates via Server-Sent Events
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import yt_dlp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported media extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg', '.3gp'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.opus'}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Import transcription components (lazy load to speed up startup)
whisper_model = None


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscriptionJob:
    id: str
    url: str
    status: JobStatus = JobStatus.QUEUED
    title: str = ""
    video_id: str = ""
    progress: float = 0.0
    message: str = "Waiting in queue..."
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None
    source_type: str = "youtube"  # "youtube" or "local"
    source_path: Optional[str] = None


# Application state
class AppState:
    def __init__(self):
        self.jobs: dict[str, TranscriptionJob] = {}
        self.queue: deque[str] = deque()
        self.processing: bool = False
        self.current_job_id: Optional[str] = None
        self.subscribers: list[asyncio.Queue] = []
        
    async def broadcast(self, event: dict):
        """Send event to all SSE subscribers"""
        for queue in self.subscribers:
            await queue.put(event)


app = FastAPI(title="YouTube & Local Media Transcriber", version="2.0.0")
state = AppState()

# Configuration from environment
MODEL_SIZE = os.environ.get("MODEL_SIZE", "large-v3")
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Max upload size (2GB)
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024


# Pydantic models
class URLSubmission(BaseModel):
    urls: str  # Newline-separated URLs


class JobResponse(BaseModel):
    id: str
    url: str
    status: str
    title: str
    progress: float
    message: str


def get_whisper_model():
    """Lazy load the Whisper model"""
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
        whisper_model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        logger.info("Model loaded successfully")
    return whisper_model


def generate_file_id(filepath: Path) -> str:
    """Generate a unique ID for a local file."""
    stat = filepath.stat()
    hash_input = f"{filepath.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def is_supported_file(filename: str) -> bool:
    """Check if file extension is supported."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def is_audio_file(filename: str) -> bool:
    """Check if file is audio."""
    ext = Path(filename).suffix.lower()
    return ext in AUDIO_EXTENSIONS


def extract_audio_from_video(video_path: Path) -> Path:
    """Extract audio from video file using ffmpeg."""
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / f"{video_path.stem}.wav"
    
    logger.info(f"Extracting audio from: {video_path.name}")
    
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        str(audio_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return audio_path


def extract_video_info(url: str) -> dict:
    """Extract video information without downloading"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'id': info.get('id', 'unknown'),
            'title': info.get('title', 'Unknown Title'),
            'duration': info.get('duration', 0),
            'channel': info.get('channel', ''),
        }


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        return output_dir / f"{video_id}.wav"


def transcribe_audio(audio_path: Path) -> dict:
    """Transcribe audio file using Whisper"""
    model = get_whisper_model()
    
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    segments_list = list(segments)
    
    result = {
        'language': info.language,
        'language_probability': info.language_probability,
        'duration': info.duration,
        'segments': []
    }
    
    for segment in segments_list:
        seg_data = {
            'id': segment.id,
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip(),
        }
        
        if segment.words:
            seg_data['words'] = [
                {
                    'word': w.word,
                    'start': w.start,
                    'end': w.end,
                    'probability': w.probability
                }
                for w in segment.words
            ]
        
        result['segments'].append(seg_data)
    
    return result


def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """Format seconds to timestamp string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def save_outputs(media_info: dict, transcript: dict, output_dir: Path) -> Path:
    """Save transcript in multiple formats"""
    media_id = media_info['id']
    title = media_info['title']
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:80]
    
    # Create output directory
    transcript_dir = output_dir / media_id
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'id': media_id,
        'title': title,
        'source_type': media_info.get('source_type', 'unknown'),
        'transcribed_at': datetime.now().isoformat(),
        'language': transcript['language'],
        'duration': transcript.get('duration'),
    }
    
    with open(transcript_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save JSON (full)
    with open(transcript_dir / f"{safe_title}.json", 'w', encoding='utf-8') as f:
        json.dump({'metadata': metadata, 'transcript': transcript}, f, indent=2, ensure_ascii=False)
    
    # Save SRT
    with open(transcript_dir / f"{safe_title}.srt", 'w', encoding='utf-8') as f:
        for i, seg in enumerate(transcript['segments'], 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'], 'srt')} --> {format_timestamp(seg['end'], 'srt')}\n")
            f.write(f"{seg['text']}\n\n")
    
    # Save VTT
    with open(transcript_dir / f"{safe_title}.vtt", 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(transcript['segments'], 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'], 'vtt')} --> {format_timestamp(seg['end'], 'vtt')}\n")
            f.write(f"{seg['text']}\n\n")
    
    # Save TXT
    with open(transcript_dir / f"{safe_title}.txt", 'w', encoding='utf-8') as f:
        for seg in transcript['segments']:
            f.write(f"[{format_timestamp(seg['start'], 'vtt')}] {seg['text']}\n")
    
    return transcript_dir


async def process_youtube_job(job_id: str):
    """Process a YouTube transcription job"""
    job = state.jobs.get(job_id)
    if not job:
        return
    
    audio_path = None
    
    try:
        # Update status: downloading
        job.status = JobStatus.DOWNLOADING
        job.message = "Extracting video information..."
        job.progress = 5
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Get video info
        video_info = extract_video_info(job.url)
        job.title = video_info['title']
        job.video_id = video_info['id']
        job.message = f"Downloading: {job.title}"
        job.progress = 10
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Check if already transcribed
        existing_dir = OUTPUT_DIR / "transcripts" / video_info['id']
        if existing_dir.exists():
            job.status = JobStatus.COMPLETED
            job.message = "Already transcribed (using cached result)"
            job.progress = 100
            job.output_dir = str(existing_dir)
            job.completed_at = datetime.now()
            await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
            return
        
        # Download audio
        audio_dir = OUTPUT_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = download_audio(job.url, audio_dir)
        job.progress = 30
        job.message = "Download complete. Starting transcription..."
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Transcribe
        job.status = JobStatus.TRANSCRIBING
        job.message = "Transcribing audio..."
        job.progress = 40
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(None, transcribe_audio, audio_path)
        
        job.progress = 85
        job.message = "Saving transcripts..."
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Save outputs
        transcript_dir = OUTPUT_DIR / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        video_info['source_type'] = 'youtube'
        output_path = save_outputs(video_info, transcript, transcript_dir)
        
        # Complete
        job.status = JobStatus.COMPLETED
        job.message = "Transcription complete!"
        job.progress = 100
        job.output_dir = str(output_path)
        job.completed_at = datetime.now()
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        logger.info(f"Completed: {job.title}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.message = f"Error: {str(e)}"
        job.error = str(e)
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
    
    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except:
                pass


async def process_local_job(job_id: str):
    """Process a local file transcription job"""
    job = state.jobs.get(job_id)
    if not job:
        return
    
    audio_path = None
    temp_audio = False
    
    try:
        source_path = Path(job.source_path)
        
        job.status = JobStatus.DOWNLOADING
        job.message = "Preparing file..."
        job.progress = 10
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Check if already transcribed
        existing_dir = OUTPUT_DIR / "transcripts" / job.video_id
        if existing_dir.exists():
            job.status = JobStatus.COMPLETED
            job.message = "Already transcribed (using cached result)"
            job.progress = 100
            job.output_dir = str(existing_dir)
            job.completed_at = datetime.now()
            await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
            return
        
        # Prepare audio
        if is_audio_file(source_path.name):
            audio_path = source_path
        else:
            job.message = "Extracting audio from video..."
            job.progress = 20
            await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
            
            loop = asyncio.get_event_loop()
            audio_path = await loop.run_in_executor(None, extract_audio_from_video, source_path)
            temp_audio = True
        
        job.progress = 30
        job.message = "Starting transcription..."
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Transcribe
        job.status = JobStatus.TRANSCRIBING
        job.message = "Transcribing audio..."
        job.progress = 40
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(None, transcribe_audio, audio_path)
        
        job.progress = 85
        job.message = "Saving transcripts..."
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        # Save outputs
        transcript_dir = OUTPUT_DIR / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        media_info = {
            'id': job.video_id,
            'title': job.title,
            'source_type': 'local',
        }
        output_path = save_outputs(media_info, transcript, transcript_dir)
        
        # Complete
        job.status = JobStatus.COMPLETED
        job.message = "Transcription complete!"
        job.progress = 100
        job.output_dir = str(output_path)
        job.completed_at = datetime.now()
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
        
        logger.info(f"Completed: {job.title}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.message = f"Error: {str(e)}"
        job.error = str(e)
        await state.broadcast({'type': 'job_update', 'job': job_to_dict(job)})
    
    finally:
        if temp_audio and audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except:
                pass


async def process_job(job_id: str):
    """Process a transcription job (routes to YouTube or local handler)"""
    job = state.jobs.get(job_id)
    if not job:
        return
    
    if job.source_type == "local":
        await process_local_job(job_id)
    else:
        await process_youtube_job(job_id)


async def process_queue():
    """Background task to process the job queue"""
    while True:
        if state.queue and not state.processing:
            state.processing = True
            job_id = state.queue.popleft()
            state.current_job_id = job_id
            
            await process_job(job_id)
            
            state.current_job_id = None
            state.processing = False
        
        await asyncio.sleep(0.5)


def job_to_dict(job: TranscriptionJob) -> dict:
    """Convert job to dictionary for JSON response"""
    return {
        'id': job.id,
        'url': job.url,
        'status': job.status.value,
        'title': job.title,
        'video_id': job.video_id,
        'progress': job.progress,
        'message': job.message,
        'created_at': job.created_at.isoformat(),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'output_dir': job.output_dir,
        'error': job.error,
        'source_type': job.source_type,
    }


# Start background queue processor
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())
    logger.info("Queue processor started")


# API Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    return get_html_template()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_SIZE, "device": DEVICE}


@app.post("/api/submit")
async def submit_urls(submission: URLSubmission):
    """Submit URLs for transcription"""
    urls = [u.strip() for u in submission.urls.strip().split('\n') if u.strip()]
    
    if not urls:
        raise HTTPException(status_code=400, detail="No valid URLs provided")
    
    # Validate URLs
    valid_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/shorts/',
    ]
    
    jobs_created = []
    
    for url in urls:
        if not any(re.search(p, url) for p in valid_patterns):
            continue
        
        job_id = str(uuid.uuid4())[:8]
        job = TranscriptionJob(id=job_id, url=url, source_type="youtube")
        state.jobs[job_id] = job
        state.queue.append(job_id)
        jobs_created.append(job_to_dict(job))
    
    if not jobs_created:
        raise HTTPException(status_code=400, detail="No valid YouTube URLs found")
    
    await state.broadcast({'type': 'jobs_added', 'jobs': jobs_created})
    
    return {"message": f"Added {len(jobs_created)} job(s) to queue", "jobs": jobs_created}


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload local media files for transcription"""
    jobs_created = []
    errors = []
    
    for file in files:
        # Validate file type
        if not is_supported_file(file.filename):
            errors.append(f"Unsupported file type: {file.filename}")
            continue
        
        try:
            # Save uploaded file
            file_id = str(uuid.uuid4())[:8]
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', file.filename)
            upload_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"
            
            with open(upload_path, 'wb') as f:
                content = await file.read()
                if len(content) > MAX_UPLOAD_SIZE:
                    errors.append(f"File too large: {file.filename} (max 2GB)")
                    continue
                f.write(content)
            
            # Create job
            title = Path(file.filename).stem
            job = TranscriptionJob(
                id=file_id,
                url=f"local://{safe_filename}",
                title=title,
                video_id=file_id,
                source_type="local",
                source_path=str(upload_path),
            )
            state.jobs[file_id] = job
            state.queue.append(file_id)
            jobs_created.append(job_to_dict(job))
            
        except Exception as e:
            errors.append(f"Failed to process {file.filename}: {str(e)}")
    
    if not jobs_created and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    
    await state.broadcast({'type': 'jobs_added', 'jobs': jobs_created})
    
    result = {"message": f"Added {len(jobs_created)} file(s) to queue", "jobs": jobs_created}
    if errors:
        result["errors"] = errors
    
    return result


@app.get("/api/jobs")
async def get_jobs():
    """Get all jobs"""
    jobs = [job_to_dict(job) for job in state.jobs.values()]
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    return {"jobs": jobs, "queue_length": len(state.queue)}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a specific job"""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_dict(job)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from the list"""
    if job_id in state.jobs:
        job = state.jobs[job_id]
        
        # Clean up uploaded file if local
        if job.source_type == "local" and job.source_path:
            try:
                Path(job.source_path).unlink(missing_ok=True)
            except:
                pass
        
        if job_id in state.queue:
            state.queue.remove(job_id)
        del state.jobs[job_id]
        await state.broadcast({'type': 'job_deleted', 'job_id': job_id})
        return {"message": "Job deleted"}
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events endpoint for real-time updates"""
    queue = asyncio.Queue()
    state.subscribers.append(queue)
    
    async def event_generator():
        try:
            jobs = [job_to_dict(job) for job in state.jobs.values()]
            yield f"data: {json.dumps({'type': 'initial', 'jobs': jobs})}\n\n"
            
            while True:
                if await request.is_disconnected():
                    break
                
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        finally:
            state.subscribers.remove(queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/files/{video_id}/{filename}")
async def download_file(video_id: str, filename: str):
    """Download a transcript file"""
    file_path = OUTPUT_DIR / "transcripts" / video_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)


@app.get("/api/transcript/{video_id}")
async def get_transcript_text(video_id: str):
    """Get the plain text transcript"""
    transcript_dir = OUTPUT_DIR / "transcripts" / video_id
    if not transcript_dir.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    txt_files = list(transcript_dir.glob("*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="Transcript file not found")
    
    with open(txt_files[0], 'r', encoding='utf-8') as f:
        return {"text": f.read(), "filename": txt_files[0].name}


@app.get("/api/formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "video": sorted(VIDEO_EXTENSIONS),
        "audio": sorted(AUDIO_EXTENSIONS),
    }


def get_html_template() -> str:
    """Return the HTML template for the web interface"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube & Media Transcriber</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #888;
            font-size: 1rem;
        }
        
        .input-section {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .input-section h2 {
            margin-bottom: 16px;
            font-size: 1.2rem;
            color: #00d2ff;
        }
        
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        
        .tab {
            padding: 10px 20px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: transparent;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .tab.active {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            color: #fff;
            border-color: transparent;
        }
        
        .tab:hover:not(.active) {
            border-color: #00d2ff;
            color: #00d2ff;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        textarea {
            width: 100%;
            height: 120px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 12px;
            color: #fff;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            resize: vertical;
            margin-bottom: 16px;
        }
        
        textarea:focus {
            outline: none;
            border-color: #00d2ff;
        }
        
        textarea::placeholder {
            color: #666;
        }
        
        .drop-zone {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 16px;
        }
        
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #00d2ff;
            background: rgba(0, 210, 255, 0.1);
        }
        
        .drop-zone-icon {
            font-size: 48px;
            margin-bottom: 12px;
        }
        
        .drop-zone-text {
            color: #888;
            margin-bottom: 8px;
        }
        
        .drop-zone-formats {
            font-size: 0.8rem;
            color: #666;
        }
        
        .file-input {
            display: none;
        }
        
        .selected-files {
            margin-top: 12px;
        }
        
        .selected-file {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            margin-bottom: 6px;
        }
        
        .selected-file-name {
            font-size: 0.9rem;
        }
        
        .selected-file-remove {
            background: none;
            border: none;
            color: #ff4757;
            cursor: pointer;
            font-size: 1.2rem;
        }
        
        .btn {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            border: none;
            padding: 12px 32px;
            border-radius: 8px;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 210, 255, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .stats {
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .stat-card {
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00d2ff;
        }
        
        .stat-label {
            font-size: 0.85rem;
            color: #888;
            margin-top: 4px;
        }
        
        .jobs-section h2 {
            margin-bottom: 16px;
            font-size: 1.2rem;
        }
        
        .job-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .job-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: border-color 0.3s;
        }
        
        .job-card.processing {
            border-color: #00d2ff;
        }
        
        .job-card.completed {
            border-color: #00ff88;
        }
        
        .job-card.failed {
            border-color: #ff4757;
        }
        
        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        
        .job-title {
            font-weight: 600;
            font-size: 1rem;
            color: #fff;
            word-break: break-word;
            flex: 1;
            margin-right: 12px;
        }
        
        .job-badges {
            display: flex;
            gap: 6px;
        }
        
        .job-status, .job-source {
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            text-transform: uppercase;
            white-space: nowrap;
        }
        
        .status-queued { background: #666; }
        .status-downloading { background: #f39c12; }
        .status-transcribing { background: #3498db; }
        .status-completed { background: #27ae60; }
        .status-failed { background: #e74c3c; }
        
        .source-youtube { background: #ff0000; }
        .source-local { background: #9b59b6; }
        
        .job-url {
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 12px;
            word-break: break-all;
        }
        
        .progress-bar {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            transition: width 0.3s ease;
        }
        
        .job-message {
            font-size: 0.85rem;
            color: #aaa;
        }
        
        .job-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        
        .job-actions a, .job-actions button {
            font-size: 0.8rem;
            padding: 6px 12px;
            border-radius: 4px;
            text-decoration: none;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: none;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .job-actions a:hover, .job-actions button:hover {
            background: rgba(255,255,255,0.2);
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        
        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 16px;
            opacity: 0.5;
        }
        
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            background: rgba(0,0,0,0.8);
        }
        
        .connection-status.connected { color: #00ff88; }
        .connection-status.disconnected { color: #ff4757; }
        
        @media (max-width: 600px) {
            .stats { flex-direction: column; }
            h1 { font-size: 1.8rem; }
            .tabs { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎬 Media Transcriber</h1>
            <p class="subtitle">Local AI-powered transcription • YouTube & Local Files</p>
        </header>
        
        <section class="input-section">
            <div class="tabs">
                <button class="tab active" data-tab="youtube">📺 YouTube URLs</button>
                <button class="tab" data-tab="upload">📁 Upload Files</button>
            </div>
            
            <div id="youtube-tab" class="tab-content active">
                <textarea id="urlInput" placeholder="Paste YouTube URLs here (one per line)&#10;&#10;https://www.youtube.com/watch?v=...&#10;https://youtu.be/..."></textarea>
                <button class="btn" id="submitBtn" onclick="submitUrls()">
                    Start Transcription
                </button>
            </div>
            
            <div id="upload-tab" class="tab-content">
                <div class="drop-zone" id="dropZone">
                    <div class="drop-zone-icon">📂</div>
                    <div class="drop-zone-text">Drag & drop files here or click to browse</div>
                    <div class="drop-zone-formats">MP4, MKV, AVI, MOV, MP3, WAV, M4A, and more</div>
                </div>
                <input type="file" id="fileInput" class="file-input" multiple accept=".mp4,.mkv,.avi,.mov,.wmv,.flv,.webm,.m4v,.mpeg,.mpg,.3gp,.mp3,.wav,.m4a,.aac,.ogg,.flac,.wma,.opus">
                <div class="selected-files" id="selectedFiles"></div>
                <button class="btn" id="uploadBtn" onclick="uploadFiles()" disabled>
                    Upload & Transcribe
                </button>
            </div>
        </section>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="queuedCount">0</div>
                <div class="stat-label">In Queue</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="processingCount">0</div>
                <div class="stat-label">Processing</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="completedCount">0</div>
                <div class="stat-label">Completed</div>
            </div>
        </div>
        
        <section class="jobs-section">
            <h2>📝 Transcription Jobs</h2>
            <div class="job-list" id="jobList">
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <p>No transcription jobs yet.<br>Add YouTube URLs or upload files to get started!</p>
                </div>
            </div>
        </section>
    </div>
    
    <div class="connection-status disconnected" id="connectionStatus">
        ● Connecting...
    </div>
    
    <script>
        let jobs = {};
        let eventSource = null;
        let selectedFiles = [];
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
            });
        });
        
        // File upload handling
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', () => {
            handleFiles(fileInput.files);
        });
        
        function handleFiles(files) {
            for (const file of files) {
                if (!selectedFiles.find(f => f.name === file.name)) {
                    selectedFiles.push(file);
                }
            }
            renderSelectedFiles();
        }
        
        function removeFile(index) {
            selectedFiles.splice(index, 1);
            renderSelectedFiles();
        }
        
        function renderSelectedFiles() {
            const container = document.getElementById('selectedFiles');
            const uploadBtn = document.getElementById('uploadBtn');
            
            if (selectedFiles.length === 0) {
                container.innerHTML = '';
                uploadBtn.disabled = true;
                return;
            }
            
            uploadBtn.disabled = false;
            container.innerHTML = selectedFiles.map((file, i) => `
                <div class="selected-file">
                    <span class="selected-file-name">📄 ${file.name}</span>
                    <button class="selected-file-remove" onclick="removeFile(${i})">×</button>
                </div>
            `).join('');
        }
        
        async function uploadFiles() {
            if (selectedFiles.length === 0) return;
            
            const btn = document.getElementById('uploadBtn');
            btn.disabled = true;
            btn.textContent = 'Uploading...';
            
            const formData = new FormData();
            for (const file of selectedFiles) {
                formData.append('files', file);
            }
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    selectedFiles = [];
                    renderSelectedFiles();
                    if (data.errors) {
                        alert('Some files had errors: ' + data.errors.join(', '));
                    }
                } else {
                    alert(data.detail || 'Error uploading files');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Upload & Transcribe';
            }
        }
        
        function connectSSE() {
            eventSource = new EventSource('/api/events');
            
            eventSource.onopen = () => {
                document.getElementById('connectionStatus').className = 'connection-status connected';
                document.getElementById('connectionStatus').textContent = '● Connected';
            };
            
            eventSource.onerror = () => {
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                document.getElementById('connectionStatus').textContent = '● Reconnecting...';
                setTimeout(connectSSE, 3000);
            };
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };
        }
        
        function handleEvent(data) {
            switch (data.type) {
                case 'initial':
                    jobs = {};
                    data.jobs.forEach(job => jobs[job.id] = job);
                    renderJobs();
                    break;
                case 'jobs_added':
                    data.jobs.forEach(job => jobs[job.id] = job);
                    renderJobs();
                    break;
                case 'job_update':
                    jobs[data.job.id] = data.job;
                    renderJobs();
                    break;
                case 'job_deleted':
                    delete jobs[data.job_id];
                    renderJobs();
                    break;
            }
        }
        
        function renderJobs() {
            const jobList = document.getElementById('jobList');
            const jobArray = Object.values(jobs).sort((a, b) => 
                new Date(b.created_at) - new Date(a.created_at)
            );
            
            // Update stats
            const queued = jobArray.filter(j => j.status === 'queued').length;
            const processing = jobArray.filter(j => ['downloading', 'transcribing'].includes(j.status)).length;
            const completed = jobArray.filter(j => j.status === 'completed').length;
            
            document.getElementById('queuedCount').textContent = queued;
            document.getElementById('processingCount').textContent = processing;
            document.getElementById('completedCount').textContent = completed;
            
            if (jobArray.length === 0) {
                jobList.innerHTML = `
                    <div class="empty-state">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        <p>No transcription jobs yet.<br>Add YouTube URLs or upload files to get started!</p>
                    </div>
                `;
                return;
            }
            
            jobList.innerHTML = jobArray.map(job => `
                <div class="job-card ${getStatusClass(job.status)}">
                    <div class="job-header">
                        <div class="job-title">${job.title || 'Loading...'}</div>
                        <div class="job-badges">
                            <span class="job-source source-${job.source_type || 'youtube'}">${job.source_type || 'youtube'}</span>
                            <span class="job-status status-${job.status}">${job.status}</span>
                        </div>
                    </div>
                    <div class="job-url">${job.url}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${job.progress}%"></div>
                    </div>
                    <div class="job-message">${job.message}</div>
                    ${job.status === 'completed' ? `
                        <div class="job-actions">
                            <a href="/api/files/${job.video_id}/${encodeURIComponent(getFilename(job, 'txt'))}" target="_blank">📄 TXT</a>
                            <a href="/api/files/${job.video_id}/${encodeURIComponent(getFilename(job, 'srt'))}" target="_blank">🎬 SRT</a>
                            <a href="/api/files/${job.video_id}/${encodeURIComponent(getFilename(job, 'vtt'))}" target="_blank">🌐 VTT</a>
                            <a href="/api/files/${job.video_id}/${encodeURIComponent(getFilename(job, 'json'))}" target="_blank">📊 JSON</a>
                            <button onclick="showTranscript('${job.video_id}')">👁️ View</button>
                            <button onclick="deleteJob('${job.id}')">🗑️ Remove</button>
                        </div>
                    ` : job.status === 'failed' ? `
                        <div class="job-actions">
                            <button onclick="deleteJob('${job.id}')">🗑️ Remove</button>
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }
        
        function getStatusClass(status) {
            if (['downloading', 'transcribing'].includes(status)) return 'processing';
            return status;
        }
        
        function getFilename(job, ext) {
            const safeTitle = job.title.replace(/[<>:"/\\|?*]/g, '_').substring(0, 80);
            return `${safeTitle}.${ext}`;
        }
        
        async function submitUrls() {
            const textarea = document.getElementById('urlInput');
            const btn = document.getElementById('submitBtn');
            const urls = textarea.value.trim();
            
            if (!urls) return;
            
            btn.disabled = true;
            btn.textContent = 'Submitting...';
            
            try {
                const response = await fetch('/api/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ urls })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    textarea.value = '';
                } else {
                    alert(data.detail || 'Error submitting URLs');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Start Transcription';
            }
        }
        
        async function deleteJob(jobId) {
            try {
                await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
            } catch (err) {
                console.error('Delete failed:', err);
            }
        }
        
        async function showTranscript(videoId) {
            try {
                const response = await fetch(`/api/transcript/${videoId}`);
                const data = await response.json();
                
                const win = window.open('', '_blank');
                win.document.write(`
                    <html>
                    <head><title>${data.filename}</title>
                    <style>
                        body { font-family: sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; line-height: 1.6; }
                        pre { white-space: pre-wrap; word-wrap: break-word; }
                    </style>
                    </head>
                    <body><pre>${data.text}</pre></body>
                    </html>
                `);
            } catch (err) {
                alert('Error loading transcript');
            }
        }
        
        document.getElementById('urlInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                submitUrls();
            }
        });
        
        connectSSE();
    </script>
</body>
</html>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
