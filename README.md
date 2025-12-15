# YouTube Transcriber 🎬

Local AI-powered YouTube transcription using faster-whisper. Runs entirely on your hardware with GPU acceleration.

**Two ways to use it:**
- **CLI** - Quick command-line transcription
- **Web UI** - Paste URLs, watch progress, download results

---

## Features

- 🚀 **GPU Accelerated** - CUDA support via faster-whisper (CTranslate2)
- 🎯 **High Accuracy** - Whisper large-v3 model with word-level timestamps
- 🗣️ **Speaker Diarization** - Identify who said what (optional, via whisperX)
- 📁 **Multiple Formats** - SRT, VTT, TXT, JSON output
- 🔄 **Batch Processing** - Process playlists, channels, or URL lists
- 🌐 **Web Interface** - Real-time progress with Docker deployment
- 💾 **Resume Support** - Skip already-processed videos
- 🏠 **Fully Local** - No cloud APIs, everything on your machine

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
git clone https://github.com/YOUR_USERNAME/youtube-transcriber.git
cd youtube-transcriber

# Setup
bash setup.sh
source venv/bin/activate

# Transcribe!
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID
```

### Option B: Docker Web UI

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/youtube-transcriber.git
cd youtube-transcriber

# Start
docker compose up -d

# Open http://localhost:8000
```

---

## CLI Usage

### Basic Commands

```bash
# Single video
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID

# Multiple videos
python transcribe.py URL1 URL2 URL3

# From a file (one URL per line)
python transcribe.py --file urls.txt

# Entire playlist
python transcribe.py --playlist "https://www.youtube.com/playlist?list=PLxxxxx"
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
python transcribe.py --model medium URL

# Use CPU (slower but no GPU needed)
python transcribe.py --device cpu --model small URL
```

### Speaker Diarization

Identify who said what in multi-speaker videos:

```bash
# Requires HuggingFace token
export HF_TOKEN='your_token'
python transcribe.py --diarize URL

# Specify expected speakers
python transcribe.py --diarize --min-speakers 2 --max-speakers 4 URL
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
  --no-skip          Re-process already transcribed videos
  --keep-audio       Keep downloaded audio files
  --file, -f         File containing URLs
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

1. Paste YouTube URLs (one per line) into the text box
2. Click "Start Transcription"
3. Watch real-time progress for each video
4. Download results (SRT/VTT/TXT/JSON) when complete

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
# Submit URLs
curl -X POST http://localhost:8000/api/submit \
  -H "Content-Type: application/json" \
  -d '{"urls": "https://www.youtube.com/watch?v=VIDEO_ID"}'

# List all jobs
curl http://localhost:8000/api/jobs

# Get job status
curl http://localhost:8000/api/jobs/{job_id}

# Download transcript
curl http://localhost:8000/api/files/{video_id}/filename.txt

# Health check
curl http://localhost:8000/health
```

Real-time updates via Server-Sent Events:
```javascript
const events = new EventSource('/api/events');
events.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## Output Structure

```
output/
├── audio/                          # Temp audio (deleted after processing)
└── transcripts/
    └── VIDEO_ID/
        ├── metadata.json           # Video info + settings
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

**VTT** - Web-friendly subtitles with speaker tags:
```
WEBVTT

1
00:00:00.000 --> 00:00:04.500
<v Speaker1>Hello and welcome to today's video.
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

# For web UI (optional)
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
python transcribe.py --model medium URL
python transcribe.py --compute-type int8 URL
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

---

## Tips & Tricks

### Process overnight
```bash
nohup python transcribe.py --file urls.txt > log.txt 2>&1 &
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

## License

MIT License - Use freely for personal and commercial projects.

## Credits

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [whisperX](https://github.com/m-bain/whisperx)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)
