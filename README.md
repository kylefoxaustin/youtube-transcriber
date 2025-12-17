# Media Transcriber 🎬

Local AI-powered transcription for YouTube videos and local media files. Runs entirely on your hardware with GPU acceleration.

**Two ways to use it:**
- **CLI** - Quick command-line transcription
- **Web UI** - Paste URLs or upload files, watch progress, download results

---

## Features

- 🚀 **GPU Accelerated** - CUDA support via faster-whisper (CTranslate2)
- 🎯 **High Accuracy** - Whisper large-v3 model with word-level timestamps
- 📺 **YouTube Support** - Videos, playlists, and channels
- 📁 **Local Files** - MP4, MKV, MOV, MP3, WAV, and many more formats
- 🗣️ **Speaker Diarization** - Identify who said what (optional)
- 📊 **Multiple Formats** - SRT, VTT, TXT, JSON output
- 🔄 **Batch Processing** - Process folders, playlists, or URL lists
- 🌐 **Web Interface** - Drag-and-drop uploads with real-time progress
- 💾 **Resume Support** - Skip already-processed files
- 🏠 **Fully Local** - No cloud APIs, everything on your machine

---

## Supported Formats

**Video:** MP4, MKV, AVI, MOV, WMV, FLV, WebM, M4V, MPEG, MPG, 3GP

**Audio:** MP3, WAV, M4A, AAC, OGG, FLAC, WMA, Opus

Basically, if ffmpeg can read it, we can transcribe it!

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (or CPU fallback)
- ~10GB disk space for large-v3 model
- ffmpeg
- Docker (optional, for web UI)

---

## Quick Start

### Option A: Command Line

```bash
# Clone
git clone https://github.com/kylefox1/youtube-transcriber.git
cd youtube-transcriber

# Setup
bash setup.sh
source venv/bin/activate

# Transcribe YouTube video
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID

# Transcribe local file
python transcribe.py /path/to/video.mp4

# Transcribe folder of videos
python transcribe.py --folder /path/to/videos/
```

### Option B: Docker Web UI

```bash
# Clone
git clone https://github.com/kylefox1/youtube-transcriber.git
cd youtube-transcriber

# Start
docker compose up -d

# Open http://localhost:8000
```

---

## CLI Usage

### Basic Commands

```bash
# YouTube video
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID

# Local video file
python transcribe.py /path/to/video.mp4

# Local audio file  
python transcribe.py /path/to/podcast.mp3

# Multiple files
python transcribe.py video1.mp4 video2.mov audio.mp3

# Mix of URLs and local files
python transcribe.py https://youtube.com/watch?v=xxx video.mp4

# From a text file (URLs or paths, one per line)
python transcribe.py --file inputs.txt

# YouTube playlist
python transcribe.py --playlist "https://www.youtube.com/playlist?list=PLxxxxx"

# Folder of media files
python transcribe.py --folder /path/to/videos/

# Folder recursive (include subfolders)
python transcribe.py --folder /path/to/videos/ --recursive
```

### Model Selection

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| tiny | ~1GB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ |
| base | ~1GB | ⚡⚡⚡⚡ | ★★★☆☆ |
| small | ~2GB | ⚡⚡⚡ | ★★★☆☆ |
| medium | ~5GB | ⚡⚡ | ★★★★☆ |
| large-v3 | ~10GB | ⚡ | ★★★★★ |

```bash
# Use a smaller/faster model
python transcribe.py --model medium video.mp4

# Use CPU (slower but no GPU needed)
python transcribe.py --device cpu --model small video.mp4
```

### Speaker Diarization

Identify who said what in multi-speaker content:

```bash
# Requires HuggingFace token
export HF_TOKEN='your_token'
python transcribe.py --diarize video.mp4

# Specify expected speakers
python transcribe.py --diarize --min-speakers 2 --max-speakers 4 video.mp4
```

To enable diarization:
1. Install whisperX: `pip install git+https://github.com/m-bain/whisperx.git`
2. Get a token at https://huggingface.co/settings/tokens
3. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

### All CLI Options

```bash
python transcribe.py --help

Options:
  --model, -m        Model size (tiny/base/small/medium/large-v2/large-v3)
  --device           cuda or cpu
  --compute-type     float16, float32, or int8
  --language, -l     Force language (auto-detect if not set)
  --output, -o       Output directory (default: ./output)
  --formats          Output formats (srt vtt txt json tsv)
  --diarize, -d      Enable speaker diarization
  --min-speakers     Min speakers for diarization
  --max-speakers     Max speakers for diarization
  --no-word-timestamps  Disable word-level timestamps
  --no-skip          Re-process already transcribed files
  --keep-audio       Keep extracted audio files
  --file, -f         File containing URLs/paths (one per line)
  --folder           Folder containing media files
  --recursive, -r    Process folders recursively
  --playlist, -p     YouTube playlist URL
```

### Extract URLs Helper

Grab all video URLs from a channel before transcribing:

```bash
# Extract from channel
python extract_urls.py "https://www.youtube.com/@ChannelName/videos" -o urls.txt

# Extract from playlist
python extract_urls.py "https://www.youtube.com/playlist?list=PLxxx" -o urls.txt

# Then transcribe
python transcribe.py --file urls.txt
```

---

## Docker Web UI

### Starting the Service

```bash
# Build and start
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

Open **http://localhost:8000** in your browser.

### Using the Web Interface

**YouTube Tab:**
1. Paste YouTube URLs (one per line) into the text box
2. Click "Start Transcription"

**Upload Tab:**
1. Drag & drop files or click to browse
2. Select video/audio files from your computer
3. Click "Upload & Transcribe"

Both methods show real-time progress and let you download results when complete.

### Management Script

```bash
./manage.sh start    # Start container
./manage.sh stop     # Stop container
./manage.sh restart  # Restart container
./manage.sh logs     # View live logs
./manage.sh gpu      # Verify GPU access
./manage.sh status   # Check if running
./manage.sh build    # Rebuild image
./manage.sh clean    # Remove all output
```

### Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Whisper model
MODEL_SIZE=large-v3

# Device (cuda or cpu)
DEVICE=cuda

# Compute type
COMPUTE_TYPE=float16
```

### API Endpoints

The web UI exposes a REST API:

```bash
# Submit YouTube URLs
curl -X POST http://localhost:8000/api/submit \
  -H "Content-Type: application/json" \
  -d '{"urls": "https://www.youtube.com/watch?v=VIDEO_ID"}'

# Upload local files
curl -X POST http://localhost:8000/api/upload \
  -F "files=@video.mp4" \
  -F "files=@audio.mp3"

# List all jobs
curl http://localhost:8000/api/jobs

# Get job status
curl http://localhost:8000/api/jobs/{job_id}

# Download transcript
curl http://localhost:8000/api/files/{video_id}/filename.txt

# Get supported formats
curl http://localhost:8000/api/formats

# Health check
curl http://localhost:8000/health
```

---

## Output Structure

```
output/
├── audio/                          # Temp audio (deleted after processing)
└── transcripts/
    └── MEDIA_ID/
        ├── metadata.json           # File info + settings
        ├── Video_Title.srt         # SubRip subtitles
        ├── Video_Title.vtt         # WebVTT subtitles
        ├── Video_Title.txt         # Plain text + timestamps
        └── Video_Title.json        # Full data + word timestamps
```

### Output Formats

**SRT** - Standard subtitle format for video players:
```
1
00:00:00,000 --> 00:00:04,500
Hello and welcome to today's video.
```

**VTT** - Web-friendly subtitles:
```
WEBVTT

1
00:00:00.000 --> 00:00:04.500
Hello and welcome to today's video.
```

**TXT** - Readable plain text:
```
[00:00:00.000] Hello and welcome to today's video.
```

**JSON** - Full data with word-level timestamps:
```json
{
  "segments": [{
    "start": 0.0,
    "end": 4.5,
    "text": "Hello and welcome",
    "words": [{"word": "Hello", "start": 0.0, "end": 0.5}, ...]
  }]
}
```

---

## Installation Details

### Manual Setup (without setup.sh)

```bash
# System dependencies
sudo apt update
sudo apt install ffmpeg python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install faster-whisper yt-dlp

# For web UI
pip install fastapi uvicorn python-multipart

# For diarization (optional)
pip install git+https://github.com/m-bain/whisperx.git
```

### cuDNN Setup

If you get `Unable to load libcudnn_ops.so`:

```bash
# Install cuDNN
sudo apt install libcudnn9-cuda-12
sudo ldconfig

# Verify
ldconfig -p | grep cudnn
```

### Docker Prerequisites

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Performance

**RTX 5090 + large-v3:**
- ~15x realtime (1 hour video ≈ 4 minutes)

**RTX 3080 + large-v3:**
- ~5x realtime (1 hour video ≈ 12 minutes)

**CPU + small model:**
- ~0.5x realtime (1 hour video ≈ 2 hours)

---

## Troubleshooting

### "CUDA out of memory"
Use a smaller model or int8 quantization:
```bash
python transcribe.py --model medium video.mp4
python transcribe.py --compute-type int8 video.mp4
```

### "ffmpeg not found"
```bash
sudo apt install ffmpeg
```

### "Unable to load libcudnn"
```bash
sudo apt install libcudnn9-cuda-12
sudo ldconfig
```

### yt-dlp JavaScript warnings
These are harmless. To silence:
```bash
sudo apt install deno
```

### Docker GPU not working
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Uploaded files not processing
Check that the uploads directory has proper permissions:
```bash
docker compose exec transcriber ls -la /app/uploads
```

---

## Tips & Tricks

### Transcribe Google Photos videos
1. Download videos from Google Photos (or use Google Takeout for bulk export)
2. Either upload via web UI or use CLI:
```bash
python transcribe.py --folder ~/Downloads/google-photos-export/
```

### Process overnight
```bash
nohup python transcribe.py --folder ~/Videos/ > log.txt 2>&1 &
tail -f transcription.log
```

### Search transcripts
```bash
grep -r "keyword" output/transcripts/
```

### Extract channel URLs
```bash
yt-dlp --flat-playlist --print url \
  "https://www.youtube.com/@ChannelName/videos" > channel_urls.txt
```

---

## Author

**Kyle Fox** - [GitHub](https://github.com/kylefox1)

## License

MIT License - Use freely for personal and commercial projects.

## Credits

Built with these excellent open-source projects:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [whisperX](https://github.com/m-bain/whisperx)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)
