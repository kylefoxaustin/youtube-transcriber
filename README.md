# YouTube Video Transcriber

A comprehensive local transcription pipeline for YouTube videos using faster-whisper. Runs entirely on your own hardware with GPU acceleration.

## Features

- **GPU-accelerated transcription** using faster-whisper (CTranslate2)
- **Speaker diarization** - identify who said what (optional, via whisperX)
- **Word-level timestamps** for precise alignment
- **Multiple output formats**: SRT, VTT, TXT, JSON, TSV
- **Batch processing** from URLs, files, or playlists
- **Resume capability** - skip already processed videos
- **Organized output** - each video gets its own folder with metadata

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- ~10GB disk space for the large-v3 model
- ffmpeg (for audio extraction)

## Installation

### 1. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg python3-pip python3-venv

# Verify ffmpeg
ffmpeg -version
```

### 2. Create virtual environment (recommended)

```bash
cd youtube-transcriber
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python packages

```bash
# Core packages
pip install faster-whisper yt-dlp

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. (Optional) Install speaker diarization

Speaker diarization requires whisperX and a HuggingFace token:

```bash
# Install whisperX
pip install git+https://github.com/m-bain/whisperx.git

# Get a HuggingFace token (free):
# 1. Create account at https://huggingface.co
# 2. Go to https://huggingface.co/settings/tokens
# 3. Create a token with read access
# 4. Accept the pyannote model terms:
#    - https://huggingface.co/pyannote/speaker-diarization-3.1
#    - https://huggingface.co/pyannote/segmentation-3.0

# Set your token
export HF_TOKEN='hf_your_token_here'

# Or add to ~/.bashrc for persistence
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

## Usage

### Basic Usage

```bash
# Single video
python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID

# Multiple videos
python transcribe.py URL1 URL2 URL3
```

### From a File

Create a text file with one URL per line:

```bash
# urls.txt
https://www.youtube.com/watch?v=abc123
https://www.youtube.com/watch?v=def456
# Comments are ignored
https://www.youtube.com/watch?v=ghi789
```

Then run:

```bash
python transcribe.py --file urls.txt
```

### Process a Playlist

```bash
python transcribe.py --playlist "https://www.youtube.com/playlist?list=PLxxxxxxxx"
```

### With Speaker Diarization

```bash
export HF_TOKEN='your_token'
python transcribe.py --diarize https://www.youtube.com/watch?v=VIDEO_ID

# Specify expected number of speakers
python transcribe.py --diarize --min-speakers 2 --max-speakers 4 URL
```

### Model Selection

| Model | Size | VRAM | Speed | Accuracy |
|-------|------|------|-------|----------|
| tiny | 75MB | ~1GB | Fastest | Lower |
| base | 142MB | ~1GB | Fast | OK |
| small | 466MB | ~2GB | Medium | Good |
| medium | 1.5GB | ~5GB | Slower | Better |
| large-v3 | 3GB | ~10GB | Slowest | Best |

```bash
# Use a smaller model for faster processing
python transcribe.py --model medium URL

# Use CPU if no GPU available (slow!)
python transcribe.py --model small --device cpu URL
```

### Output Formats

```bash
# Default: srt, vtt, txt, json
python transcribe.py URL

# Custom formats
python transcribe.py --formats srt txt URL

# All formats including TSV
python transcribe.py --formats srt vtt txt json tsv URL
```

### Other Options

```bash
# Custom output directory
python transcribe.py --output ~/my_transcripts URL

# Force specific language (skip auto-detect)
python transcribe.py --language en URL

# Re-process already transcribed videos
python transcribe.py --no-skip URL

# Keep downloaded audio files
python transcribe.py --keep-audio URL

# Disable word-level timestamps (faster)
python transcribe.py --no-word-timestamps URL
```

## Output Structure

```
output/
├── audio/                          # Temporary audio (deleted unless --keep-audio)
├── transcripts/
│   └── VIDEO_ID/
│       ├── metadata.json           # Video info + transcription settings
│       ├── Video_Title.srt         # SubRip subtitles
│       ├── Video_Title.vtt         # WebVTT subtitles
│       ├── Video_Title.txt         # Plain text with timestamps
│       └── Video_Title.json        # Full transcript with word timestamps
└── transcription.log               # Processing log
```

## Output Format Details

### SRT (SubRip)
Standard subtitle format, works with most video players:
```
1
00:00:00,000 --> 00:00:04,500
[SPEAKER_00] Hello and welcome to today's video.

2
00:00:04,500 --> 00:00:08,200
[SPEAKER_01] Thanks for having me!
```

### VTT (WebVTT)
Web-friendly subtitle format with speaker tags:
```
WEBVTT

1
00:00:00.000 --> 00:00:04.500
<v SPEAKER_00>Hello and welcome to today's video.

2
00:00:04.500 --> 00:00:08.200
<v SPEAKER_01>Thanks for having me!
```

### TXT (Plain Text)
Easy to read, searchable:
```
SPEAKER_00:
[00:00:00.000] Hello and welcome to today's video.

SPEAKER_01:
[00:00:04.500] Thanks for having me!
```

### JSON (Full Detail)
Complete data including word-level timestamps:
```json
{
  "metadata": {
    "video_id": "abc123",
    "title": "My Video",
    "language": "en",
    ...
  },
  "transcript": {
    "segments": [
      {
        "start": 0.0,
        "end": 4.5,
        "text": "Hello and welcome",
        "speaker": "SPEAKER_00",
        "words": [
          {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
          ...
        ]
      }
    ]
  }
}
```

### TSV (Tab-Separated)
Import into spreadsheets:
```
start	end	speaker	text
0.000	4.500	SPEAKER_00	Hello and welcome to today's video.
4.500	8.200	SPEAKER_01	Thanks for having me!
```

## Tips & Tricks

### Processing Large Batches

For hundreds of videos, consider running overnight:
```bash
nohup python transcribe.py --file urls.txt > transcribe_output.log 2>&1 &

# Monitor progress
tail -f transcription.log
```

### Extracting URLs from YouTube

Get all video URLs from a channel using yt-dlp:
```bash
# List all videos from a channel (don't download)
yt-dlp --flat-playlist --print url "https://www.youtube.com/@ChannelName/videos" > channel_urls.txt

# Then transcribe
python transcribe.py --file channel_urls.txt
```

### VRAM Management

If you run out of VRAM:
```bash
# Use a smaller model
python transcribe.py --model medium URL

# Or use int8 quantization
python transcribe.py --compute-type int8 URL

# Or fall back to CPU (slow)
python transcribe.py --device cpu --model small URL
```

### Combining with Other Tools

```bash
# Search transcripts
grep -r "keyword" output/transcripts/*/

# Convert JSON to other formats with jq
cat output/transcripts/*/Video.json | jq '.transcript.segments[].text'
```

## Troubleshooting

### "CUDA out of memory"
- Use a smaller model: `--model medium` or `--model small`
- Use int8 quantization: `--compute-type int8`
- Close other GPU applications

### "ffmpeg not found"
```bash
sudo apt install ffmpeg
```

### "Could not load library cudnn_ops_infer64_8.dll"
Your CUDA/cuDNN setup may be incomplete. Verify:
```bash
nvidia-smi  # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Diarization not working
1. Ensure whisperX is installed: `pip install git+https://github.com/m-bain/whisperx.git`
2. Set HF_TOKEN: `export HF_TOKEN='your_token'`
3. Accept model terms on HuggingFace (see installation section)

### Slow downloads
YouTube may throttle. Try:
```bash
# Use aria2 for faster downloads (if installed)
pip install yt-dlp[aria2]
```

## Performance Benchmarks

On RTX 5090 with large-v3 model:
- ~10x realtime speed (1 hour video ≈ 6 minutes)
- Add ~20% time for diarization

On RTX 3080 with large-v3 model:
- ~5x realtime speed (1 hour video ≈ 12 minutes)

## License

MIT License - Use freely for personal and commercial projects.

## Credits

Built on these excellent open-source projects:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [whisperX](https://github.com/m-bain/whisperx)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [OpenAI Whisper](https://github.com/openai/whisper)
