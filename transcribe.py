#!/usr/bin/env python3
"""
YouTube Video Transcriber
A comprehensive local transcription pipeline using faster-whisper and whisperX.

Features:
- GPU-accelerated transcription (CUDA)
- Speaker diarization (who said what)
- Word-level timestamps
- Multiple output formats (SRT, VTT, TXT, JSON, TSV)
- Batch processing from URLs or playlists
- Resume capability (skip already processed)
- Organized output structure
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Check for required packages and provide helpful error messages
def check_dependencies():
    missing = []
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    try:
        import yt_dlp
    except ImportError:
        missing.append("yt-dlp")
    
    if missing:
        print("Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from faster_whisper import WhisperModel
import yt_dlp

# Optional: whisperX for speaker diarization
try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transcription.log')
    ]
)
logger = logging.getLogger(__name__)


class TranscriptionConfig:
    """Configuration for transcription pipeline."""
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
        output_dir: str = "output",
        output_formats: list = None,
        word_timestamps: bool = True,
        diarize: bool = False,
        hf_token: Optional[str] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        skip_existing: bool = True,
        keep_audio: bool = False,
        audio_format: str = "wav",
        batch_size: int = 16,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_formats = output_formats or ["srt", "vtt", "txt", "json"]
        self.word_timestamps = word_timestamps
        self.diarize = diarize
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.skip_existing = skip_existing
        self.keep_audio = keep_audio
        self.audio_format = audio_format
        self.batch_size = batch_size
        
        # Validate diarization requirements
        if self.diarize and not WHISPERX_AVAILABLE:
            logger.warning("whisperX not installed. Diarization disabled.")
            logger.warning("Install with: pip install whisperx")
            self.diarize = False
        
        if self.diarize and not self.hf_token:
            logger.warning("HF_TOKEN not set. Diarization requires a HuggingFace token.")
            logger.warning("Get one at: https://huggingface.co/settings/tokens")
            logger.warning("Then: export HF_TOKEN='your_token_here'")
            self.diarize = False


class YouTubeTranscriber:
    """Main transcription pipeline."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.model = None
        self.diarize_model = None
        self.align_model = None
        self.align_metadata = None
        
        # Create output directory structure
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "audio").mkdir(exist_ok=True)
        (self.config.output_dir / "transcripts").mkdir(exist_ok=True)
        
    def load_models(self):
        """Load transcription models."""
        logger.info(f"Loading Whisper model: {self.config.model_size}")
        logger.info(f"Device: {self.config.device}, Compute type: {self.config.compute_type}")
        
        self.model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type
        )
        
        if self.config.diarize and WHISPERX_AVAILABLE:
            logger.info("Loading diarization model...")
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.config.hf_token,
                device=self.config.device
            )
    
    def download_video(self, url: str) -> Optional[dict]:
        """Download video and extract audio."""
        audio_dir = self.config.output_dir / "audio"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.config.audio_format,
                'preferredquality': '192',
            }],
            'outtmpl': str(audio_dir / '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                video_id = info.get('id', 'unknown')
                title = info.get('title', 'Unknown Title')
                
                # Clean title for filesystem
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]
                
                audio_path = audio_dir / f"{video_id}.{self.config.audio_format}"
                
                # Check if already processed
                transcript_dir = self.config.output_dir / "transcripts" / video_id
                if self.config.skip_existing and transcript_dir.exists():
                    logger.info(f"Skipping (already processed): {title}")
                    return None
                
                # Download
                logger.info(f"Downloading: {title}")
                ydl.download([url])
                
                return {
                    'id': video_id,
                    'title': title,
                    'safe_title': safe_title,
                    'audio_path': audio_path,
                    'url': url,
                    'duration': info.get('duration'),
                    'upload_date': info.get('upload_date'),
                    'channel': info.get('channel'),
                    'description': info.get('description', '')[:500],
                }
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> dict:
        """Transcribe audio file."""
        logger.info(f"Transcribing: {audio_path.name}")
        
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language or self.config.language,
            word_timestamps=self.config.word_timestamps,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # Convert generator to list
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
            
            if self.config.word_timestamps and segment.words:
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
    
    def diarize_transcript(self, audio_path: Path, transcript: dict) -> dict:
        """Add speaker labels to transcript using whisperX."""
        if not self.config.diarize or not WHISPERX_AVAILABLE:
            return transcript
        
        logger.info("Running speaker diarization...")
        
        try:
            # Load audio for alignment
            audio = whisperx.load_audio(str(audio_path))
            
            # Load alignment model if needed
            if self.align_model is None:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=transcript['language'],
                    device=self.config.device
                )
            
            # Prepare segments for whisperX format
            whisperx_segments = [
                {'start': s['start'], 'end': s['end'], 'text': s['text']}
                for s in transcript['segments']
            ]
            
            # Align
            aligned = whisperx.align(
                whisperx_segments,
                self.align_model,
                self.align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False
            )
            
            # Diarize
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            
            # Assign speakers
            result = whisperx.assign_word_speakers(diarize_segments, aligned)
            
            # Update transcript with speaker info
            for i, seg in enumerate(result.get('segments', [])):
                if i < len(transcript['segments']):
                    transcript['segments'][i]['speaker'] = seg.get('speaker', 'UNKNOWN')
            
            # Count unique speakers
            speakers = set(s.get('speaker') for s in transcript['segments'] if s.get('speaker'))
            transcript['speakers'] = list(speakers)
            logger.info(f"Identified {len(speakers)} speakers")
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            logger.info("Continuing without speaker labels...")
        
        return transcript
    
    def format_timestamp(self, seconds: float, format_type: str = "srt") -> str:
        """Format seconds to timestamp string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if format_type == "srt":
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
        else:  # vtt
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def save_outputs(self, video_info: dict, transcript: dict):
        """Save transcript in all requested formats."""
        video_id = video_info['id']
        safe_title = video_info['safe_title']
        
        # Create video-specific directory
        output_dir = self.config.output_dir / "transcripts" / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'video_id': video_id,
            'title': video_info['title'],
            'url': video_info['url'],
            'channel': video_info.get('channel'),
            'duration': video_info.get('duration'),
            'upload_date': video_info.get('upload_date'),
            'transcribed_at': datetime.now().isoformat(),
            'language': transcript['language'],
            'language_probability': transcript['language_probability'],
            'speakers': transcript.get('speakers', []),
            'model': self.config.model_size,
        }
        
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Generate each format
        for fmt in self.config.output_formats:
            output_path = output_dir / f"{safe_title}.{fmt}"
            
            if fmt == "json":
                self._save_json(output_path, transcript, metadata)
            elif fmt == "srt":
                self._save_srt(output_path, transcript)
            elif fmt == "vtt":
                self._save_vtt(output_path, transcript)
            elif fmt == "txt":
                self._save_txt(output_path, transcript)
            elif fmt == "tsv":
                self._save_tsv(output_path, transcript)
            
            logger.info(f"Saved: {output_path.name}")
    
    def _save_json(self, path: Path, transcript: dict, metadata: dict):
        """Save as JSON with full detail."""
        output = {
            'metadata': metadata,
            'transcript': transcript
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _save_srt(self, path: Path, transcript: dict):
        """Save as SRT subtitle file."""
        with open(path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(transcript['segments'], 1):
                speaker = f"[{seg.get('speaker', '')}] " if seg.get('speaker') else ""
                f.write(f"{i}\n")
                f.write(f"{self.format_timestamp(seg['start'], 'srt')} --> {self.format_timestamp(seg['end'], 'srt')}\n")
                f.write(f"{speaker}{seg['text']}\n\n")
    
    def _save_vtt(self, path: Path, transcript: dict):
        """Save as WebVTT subtitle file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(transcript['segments'], 1):
                speaker = f"<v {seg.get('speaker', 'Speaker')}>" if seg.get('speaker') else ""
                f.write(f"{i}\n")
                f.write(f"{self.format_timestamp(seg['start'], 'vtt')} --> {self.format_timestamp(seg['end'], 'vtt')}\n")
                f.write(f"{speaker}{seg['text']}\n\n")
    
    def _save_txt(self, path: Path, transcript: dict):
        """Save as plain text with timestamps."""
        with open(path, 'w', encoding='utf-8') as f:
            current_speaker = None
            for seg in transcript['segments']:
                speaker = seg.get('speaker')
                timestamp = f"[{self.format_timestamp(seg['start'], 'vtt')}]"
                
                if speaker and speaker != current_speaker:
                    f.write(f"\n{speaker}:\n")
                    current_speaker = speaker
                
                f.write(f"{timestamp} {seg['text']}\n")
    
    def _save_tsv(self, path: Path, transcript: dict):
        """Save as TSV for spreadsheet import."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("start\tend\tspeaker\ttext\n")
            for seg in transcript['segments']:
                speaker = seg.get('speaker', '')
                # Escape tabs and newlines in text
                text = seg['text'].replace('\t', ' ').replace('\n', ' ')
                f.write(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{speaker}\t{text}\n")
    
    def process_video(self, url: str) -> bool:
        """Process a single video URL."""
        # Download
        video_info = self.download_video(url)
        if video_info is None:
            return False
        
        audio_path = video_info['audio_path']
        
        try:
            # Transcribe
            transcript = self.transcribe(audio_path)
            
            # Diarize if enabled
            transcript = self.diarize_transcript(audio_path, transcript)
            
            # Save outputs
            self.save_outputs(video_info, transcript)
            
            logger.info(f"✓ Completed: {video_info['title']}")
            return True
            
        finally:
            # Cleanup audio if not keeping
            if not self.config.keep_audio and audio_path.exists():
                audio_path.unlink()
    
    def process_urls(self, urls: list):
        """Process multiple URLs."""
        if not self.model:
            self.load_models()
        
        total = len(urls)
        success = 0
        failed = []
        
        for i, url in enumerate(urls, 1):
            logger.info(f"\n[{i}/{total}] Processing: {url}")
            try:
                if self.process_video(url):
                    success += 1
            except Exception as e:
                logger.error(f"Failed: {e}")
                failed.append(url)
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Completed: {success}/{total}")
        if failed:
            logger.info(f"Failed URLs:")
            for url in failed:
                logger.info(f"  - {url}")
    
    def process_playlist(self, playlist_url: str):
        """Process all videos in a YouTube playlist."""
        logger.info(f"Extracting playlist: {playlist_url}")
        
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            
            if 'entries' not in info:
                logger.error("No videos found in playlist")
                return
            
            urls = [
                f"https://www.youtube.com/watch?v={entry['id']}"
                for entry in info['entries']
                if entry and entry.get('id')
            ]
            
            logger.info(f"Found {len(urls)} videos in playlist")
            self.process_urls(urls)


def load_urls_from_file(filepath: str) -> list:
    """Load URLs from a text file (one per line)."""
    urls = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos locally using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python transcribe.py https://www.youtube.com/watch?v=VIDEO_ID

  # Multiple videos
  python transcribe.py URL1 URL2 URL3

  # From file (one URL per line)
  python transcribe.py --file urls.txt

  # Playlist
  python transcribe.py --playlist https://www.youtube.com/playlist?list=PLAYLIST_ID

  # With speaker diarization (requires HF_TOKEN)
  export HF_TOKEN='your_token'
  python transcribe.py --diarize https://www.youtube.com/watch?v=VIDEO_ID

  # Custom model and output
  python transcribe.py --model medium --output ./my_transcripts URL
        """
    )
    
    # Input options
    parser.add_argument('urls', nargs='*', help='YouTube video URLs')
    parser.add_argument('--file', '-f', help='File containing URLs (one per line)')
    parser.add_argument('--playlist', '-p', help='YouTube playlist URL')
    
    # Model options
    parser.add_argument('--model', '-m', default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                        help='Whisper model size (default: large-v3)')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--compute-type', default='float16',
                        choices=['float16', 'float32', 'int8'],
                        help='Compute type (default: float16)')
    parser.add_argument('--language', '-l',
                        help='Force language (auto-detect if not specified)')
    
    # Output options
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--formats', nargs='+', default=['srt', 'vtt', 'txt', 'json'],
                        choices=['srt', 'vtt', 'txt', 'json', 'tsv'],
                        help='Output formats (default: srt vtt txt json)')
    
    # Feature options
    parser.add_argument('--diarize', '-d', action='store_true',
                        help='Enable speaker diarization (requires whisperX and HF_TOKEN)')
    parser.add_argument('--min-speakers', type=int, default=1,
                        help='Minimum speakers for diarization (default: 1)')
    parser.add_argument('--max-speakers', type=int, default=10,
                        help='Maximum speakers for diarization (default: 10)')
    parser.add_argument('--no-word-timestamps', action='store_true',
                        help='Disable word-level timestamps')
    
    # Processing options
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-process already transcribed videos')
    parser.add_argument('--keep-audio', action='store_true',
                        help='Keep downloaded audio files')
    
    args = parser.parse_args()
    
    # Collect URLs
    urls = list(args.urls) if args.urls else []
    
    if args.file:
        urls.extend(load_urls_from_file(args.file))
    
    if not urls and not args.playlist:
        parser.print_help()
        print("\nError: No URLs provided. Use positional args, --file, or --playlist")
        sys.exit(1)
    
    # Create config
    config = TranscriptionConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        output_dir=args.output,
        output_formats=args.formats,
        word_timestamps=not args.no_word_timestamps,
        diarize=args.diarize,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        skip_existing=not args.no_skip,
        keep_audio=args.keep_audio,
    )
    
    # Create transcriber and process
    transcriber = YouTubeTranscriber(config)
    
    if args.playlist:
        transcriber.process_playlist(args.playlist)
    
    if urls:
        transcriber.process_urls(urls)


if __name__ == "__main__":
    main()
