# Media Transcriber 🎬

Local AI-powered transcription for YouTube videos and local media files. Runs entirely on your hardware with GPU acceleration.

**v1.1.0 NEW: Speaker Identification** — Create a voice profile and extract only YOUR voice from videos!

---

## Features

- 🚀 **GPU Accelerated** - CUDA support via faster-whisper (CTranslate2)
- 🎯 **High Accuracy** - Whisper large-v3 model with word-level timestamps
- 📺 **YouTube Support** - Videos, playlists, and channels
- 📁 **Local Files** - MP4, MKV, MOV, MP3, WAV, and many more formats
- 🎤 **Speaker Identification** - Extract only YOUR voice from multi-speaker videos *(NEW in v1.1.0)*
- 🗣️ **Speaker Diarization** - Identify who said what (optional)
- 📊 **Multiple Formats** - SRT, VTT, TXT, JSON output
- 🔄 **Batch Processing** - Process folders, playlists, or URL lists
- 🌐 **Web Interface** - Drag-and-drop uploads with real-time progress
- 💾 **Resume Support** - Skip already-processed files
- 🏠 **Fully Local** - No cloud APIs, everything on your machine

---

## What's New in v1.1.0 🎤

**Speaker Identification** lets you:
- Create a "voice profile" from a sample of you speaking
- Automatically find and extract YOUR segments from any video
- Perfect for pulling your voice out of family videos, meetings, podcasts

```bash
# One-time: Create your voice profile
python voice_profile.py create kyle ~/Videos/me_talking.mp4

# Extract your voice from any video
python extract_speaker.py kyle family_reunion.mp4

# Output: Just your text, SRT subtitles, or audio-only file
```

---

## Quick Start

### Option A: Docker Web UI

```bash
git clone https://github.com/kylefoxaustin/media-transcriber.git
cd media-transcriber
docker compose up -d
# Open http://localhost:8000
```

### Option B: Command Line

```bash
git clone https://github.com/kylefoxaustin/media-transcriber.git
cd media-transcriber
bash setup.sh
source venv/bin/activate

# Transcribe
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID
python transcribe.py /path/to/video.mp4
```

---

## Speaker Identification (v1.1.0)

### Step 1: Create Your Voice Profile

You need a sample where **only you** are speaking (30-60 seconds is ideal):

```bash
# From a video file
python voice_profile.py create kyle ~/Videos/me_presenting.mp4

# From a specific time range (you at 0:30-1:30)
python voice_profile.py create kyle video.mp4 --start 30 --end 90

# From an audio file
python voice_profile.py create kyle my_voice_memo.m4a
```

### Step 2: Extract Your Voice from Any Video

```bash
# Extract your segments (outputs text, SRT, audio, JSON)
python extract_speaker.py kyle family_video.mp4

# Just get the text
python extract_speaker.py kyle video.mp4 --output-text

# Just get subtitles of your parts
python extract_speaker.py kyle video.mp4 --output-srt

# Just get audio of you speaking
python extract_speaker.py kyle video.mp4 --output-audio

# Lower threshold = more matches (may include false positives)
python extract_speaker.py kyle video.mp4 --threshold 0.5
```

### Voice Profile Management

```bash
# List all profiles
python voice_profile.py list

# Test if a file matches your voice
python voice_profile.py test kyle unknown_video.mp4

# Delete a profile
python voice_profile.py delete kyle

# View profile details
python voice_profile.py info kyle
```

### How It Works

1. **Voice Embedding**: Uses SpeechBrain's ECAPA-TDNN model to create a 192-dimensional "voiceprint" from your sample
2. **Transcription**: Whisper transcribes the target video into segments
3. **Matching**: Each segment's audio is compared to your voiceprint using cosine similarity
4. **Extraction**: Segments above the threshold (default 0.6) are extracted

### Tips for Best Results

- **Good samples**: Clear audio, just your voice, minimal background noise
- **30-60 seconds**: Enough to capture your voice characteristics
- **Multiple samples**: You can add more samples to improve accuracy:
  ```bash
  python voice_profile.py create kyle sample1.wav sample2.wav sample3.wav
  ```
- **Threshold tuning**: Start at 0.6, lower to 0.5 if missing segments, raise to 0.7 if getting false positives

---

## Basic Transcription

### CLI Usage

```bash
# YouTube video
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID

# Local video file
python transcribe.py /path/to/video.mp4

# Local audio file  
python transcribe.py /path/to/podcast.mp3

# Multiple files
python transcribe.py video1.mp4 video2.mov audio.mp3

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

---

## Docker Web UI

### Starting the Service

```bash
docker compose up -d
# Open http://localhost:8000
```

### Features
- **YouTube Tab**: Paste URLs, watch real-time progress
- **Upload Tab**: Drag-and-drop local files
- Download results in SRT, VTT, TXT, JSON formats

### Management

```bash
./manage.sh start    # Start container
./manage.sh stop     # Stop container
./manage.sh logs     # View live logs
./manage.sh gpu      # Verify GPU access
```

---

## Output Structure

```
output/
├── transcripts/
│   └── VIDEO_ID/
│       ├── metadata.json
│       ├── Video_Title.srt
│       ├── Video_Title.vtt
│       ├── Video_Title.txt
│       └── Video_Title.json
│
└── speaker_extract/           # Speaker ID outputs
    ├── video_kyle.txt         # Just your text
    ├── video_kyle.srt         # Your subtitles
    ├── video_kyle.wav         # Audio of just you
    └── video_kyle.json        # Full data with confidence scores
```

---

## Supported Formats

**Video:** MP4, MKV, AVI, MOV, WMV, FLV, WebM, M4V, MPEG, MPG, 3GP

**Audio:** MP3, WAV, M4A, AAC, OGG, FLAC, WMA, Opus

---

## Installation

### Quick Setup

```bash
git clone https://github.com/kylefoxaustin/media-transcriber.git
cd media-transcriber
bash setup.sh
source venv/bin/activate
```

### Manual Setup

```bash
# System dependencies
sudo apt update
sudo apt install ffmpeg python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### cuDNN Setup (if needed)

If you get `Unable to load libcudnn_ops.so`:

```bash
sudo apt install libcudnn9-cuda-12
sudo ldconfig
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (or CPU fallback)
- ~10GB disk space for Whisper large-v3 model
- ~1GB for speaker recognition model
- ffmpeg
- Docker (optional, for web UI)

---

## Performance

**RTX 5090 + large-v3:**
- Transcription: ~15x realtime (1 hour video ≈ 4 minutes)
- Speaker ID: ~30 segments/minute

**RTX 3080 + large-v3:**
- Transcription: ~5x realtime (1 hour video ≈ 12 minutes)

---

## Troubleshooting

### "CUDA out of memory"
```bash
python transcribe.py --model medium video.mp4
```

### "SpeechBrain not installed"
```bash
pip install speechbrain torchaudio
```

### Low speaker match accuracy
- Use a cleaner voice sample (just you, no background noise)
- Try a longer sample (60+ seconds)
- Lower the threshold: `--threshold 0.5`

### No matches found
- Threshold might be too high: try `--threshold 0.4`
- Voice sample quality might be poor
- Background noise in target video

---

## Author

**Kyle Fox** - [GitHub](https://github.com/kylefoxaustin)

## License

MIT License - Use freely for personal and commercial projects.

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [SpeechBrain](https://speechbrain.github.io/) - Speaker verification
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Version History

- **v1.1.0** - Speaker identification: voice profiles and speaker extraction
- **v1.0.0** - Initial release: YouTube + local file transcription
