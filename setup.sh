#!/bin/bash
# Quick setup script for YouTube Transcriber

set -e

echo "====================================="
echo "YouTube Transcriber - Quick Setup"
echo "====================================="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg found"
else
    echo "✗ ffmpeg not found. Installing..."
    sudo apt update && sudo apt install -y ffmpeg
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install faster-whisper yt-dlp

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('✗ CUDA not available - will use CPU (slower)')
"

# Optional: whisperX
echo ""
read -p "Install whisperX for speaker diarization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing whisperX..."
    pip install git+https://github.com/m-bain/whisperx.git
    echo ""
    echo "NOTE: Speaker diarization requires a HuggingFace token."
    echo "1. Get a token at: https://huggingface.co/settings/tokens"
    echo "2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "3. Run: export HF_TOKEN='your_token_here'"
fi

echo ""
echo "====================================="
echo "Setup complete!"
echo "====================================="
echo ""
echo "Quick start:"
echo "  source venv/bin/activate"
echo "  python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID"
echo ""
echo "For a full playlist:"
echo "  python transcribe.py --playlist 'https://www.youtube.com/playlist?list=PLxxxxx'"
echo ""
echo "See README.md for more options."
