#!/usr/bin/env python3
"""
Speaker Extractor CLI
Extract only your voice/text from videos using voice profiles.

Usage:
    # First, create a voice profile (one time)
    python voice_profile.py create kyle ~/Videos/just_me_talking.mp4
    
    # Then extract your segments from any video
    python extract_speaker.py kyle family_video.mp4
    
    # Output options
    python extract_speaker.py kyle video.mp4 --output-text     # Just your text
    python extract_speaker.py kyle video.mp4 --output-srt      # Subtitles of your parts
    python extract_speaker.py kyle video.mp4 --output-audio    # Audio of just you
    python extract_speaker.py kyle video.mp4 --output-all      # Everything
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required packages."""
    missing = []
    try:
        import speechbrain
    except ImportError:
        missing.append("speechbrain")
    try:
        import torchaudio
    except ImportError:
        missing.append("torchaudio")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing.append("faster-whisper")
    
    if missing:
        print("Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video file."""
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ar', '16000', '-ac', '1',
        '-loglevel', 'error',
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def transcribe_audio(audio_path: str, model_size: str = "large-v3",
                     device: str = "cuda") -> dict:
    """Transcribe audio file."""
    from faster_whisper import WhisperModel
    
    logger.info(f"Loading Whisper model: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type="float16")
    
    logger.info("Transcribing audio...")
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
    )
    
    segments_list = []
    for seg in segments:
        segments_list.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text.strip(),
        })
    
    return {
        'language': info.language,
        'duration': info.duration,
        'segments': segments_list,
    }


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def save_text(segments: list, output_path: str):
    """Save just the text."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(f"{seg['text']}\n")
    logger.info(f"Saved text: {output_path}")


def save_text_with_timestamps(segments: list, output_path: str):
    """Save text with timestamps."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            mins = int(seg['start'] // 60)
            secs = seg['start'] % 60
            f.write(f"[{mins:02d}:{secs:05.2f}] {seg['text']}\n")
    logger.info(f"Saved timestamped text: {output_path}")


def save_srt(segments: list, output_path: str):
    """Save as SRT subtitles."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    logger.info(f"Saved SRT: {output_path}")


def save_json(segments: list, metadata: dict, output_path: str):
    """Save full data as JSON."""
    output = {
        'metadata': metadata,
        'segments': segments,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON: {output_path}")


def extract_audio_segments(audio_path: str, segments: list, output_path: str):
    """Extract and concatenate matching audio segments."""
    if not segments:
        logger.warning("No segments to extract")
        return
    
    # Create temp files for each segment
    temp_files = []
    concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    
    try:
        for i, seg in enumerate(segments):
            temp_seg = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_files.append(temp_seg.name)
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(seg['start']),
                '-t', str(seg['end'] - seg['start']),
                '-ar', '16000', '-ac', '1',
                '-loglevel', 'error',
                temp_seg.name
            ]
            subprocess.run(cmd, check=True)
            concat_list.write(f"file '{temp_seg.name}'\n")
        
        concat_list.close()
        
        # Concatenate all segments
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list.name,
            '-c', 'copy',
            '-loglevel', 'error',
            output_path
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Saved audio: {output_path}")
        
    finally:
        # Cleanup
        os.unlink(concat_list.name)
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific speaker's voice/text from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract your segments from a family video
  python extract_speaker.py kyle family_video.mp4
  
  # Get just the text (no timestamps)
  python extract_speaker.py kyle video.mp4 --output-text
  
  # Get subtitles of just your parts
  python extract_speaker.py kyle video.mp4 --output-srt
  
  # Extract audio of just you speaking
  python extract_speaker.py kyle video.mp4 --output-audio
  
  # Lower threshold for more matches (may include false positives)
  python extract_speaker.py kyle video.mp4 --threshold 0.5
  
  # Use a smaller/faster model
  python extract_speaker.py kyle video.mp4 --model small

First, create a voice profile:
  python voice_profile.py create kyle ~/Videos/just_me_talking.mp4
"""
    )
    
    parser.add_argument('profile', help='Voice profile name to match')
    parser.add_argument('source', help='Video or audio file to process')
    
    parser.add_argument('--output', '-o', help='Output directory (default: ./output/speaker_extract)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                        help='Match threshold 0-1 (default: 0.6, lower=more matches)')
    
    parser.add_argument('--output-text', action='store_true',
                        help='Output plain text')
    parser.add_argument('--output-srt', action='store_true',
                        help='Output SRT subtitles')
    parser.add_argument('--output-audio', action='store_true',
                        help='Output extracted audio')
    parser.add_argument('--output-json', action='store_true',
                        help='Output full JSON data')
    parser.add_argument('--output-all', action='store_true',
                        help='Output all formats')
    
    parser.add_argument('--model', '-m', default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                        help='Whisper model size (default: large-v3)')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for transcription')
    
    parser.add_argument('--profiles-dir', default='voice_profiles',
                        help='Voice profiles directory')
    
    args = parser.parse_args()
    
    # Default to all outputs if none specified
    if not any([args.output_text, args.output_srt, args.output_audio, args.output_json]):
        args.output_all = True
    
    if args.output_all:
        args.output_text = True
        args.output_srt = True
        args.output_audio = True
        args.output_json = True
    
    check_dependencies()
    
    from speaker_id import SpeakerIdentifier
    
    # Check source file
    source = Path(args.source)
    if not source.exists():
        print(f"Error: File not found: {source}")
        sys.exit(1)
    
    # Check profile
    identifier = SpeakerIdentifier(args.profiles_dir)
    if args.profile not in identifier.list_profiles():
        print(f"Error: Voice profile '{args.profile}' not found.")
        print(f"Available profiles: {identifier.list_profiles()}")
        print(f"\nCreate one with:")
        print(f"  python voice_profile.py create {args.profile} <your_voice_sample.mp4>")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output) if args.output else Path('output/speaker_extract')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = source.stem
    
    print(f"\n{'='*60}")
    print(f"Speaker Extraction")
    print(f"{'='*60}")
    print(f"Source:    {source.name}")
    print(f"Profile:   {args.profile}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*60}\n")
    
    # Step 1: Extract audio if video
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if source.suffix.lower() in video_exts:
            logger.info("Step 1/3: Extracting audio from video...")
            audio_path = os.path.join(tmpdir, 'audio.wav')
            extract_audio(str(source), audio_path)
        else:
            audio_path = str(source)
        
        # Step 2: Transcribe
        logger.info("Step 2/3: Transcribing audio...")
        transcript = transcribe_audio(audio_path, args.model, args.device)
        total_segments = len(transcript['segments'])
        logger.info(f"Found {total_segments} segments")
        
        # Step 3: Identify speaker segments
        logger.info(f"Step 3/3: Identifying '{args.profile}' segments...")
        matching = identifier.find_speaker_segments(
            args.profile,
            audio_path,
            transcript['segments'],
            threshold=args.threshold,
        )
        
        if not matching:
            print(f"\n⚠️  No segments matched profile '{args.profile}'")
            print(f"   Try lowering threshold: --threshold 0.5")
            sys.exit(0)
        
        # Calculate stats
        match_duration = sum(s['end'] - s['start'] for s in matching)
        total_duration = transcript['duration']
        
        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"Matched: {len(matching)}/{total_segments} segments ({len(matching)/total_segments:.1%})")
        print(f"Duration: {match_duration:.1f}s / {total_duration:.1f}s ({match_duration/total_duration:.1%})")
        print(f"{'='*60}\n")
        
        # Metadata for outputs
        metadata = {
            'source_file': str(source),
            'profile': args.profile,
            'threshold': args.threshold,
            'extracted_at': datetime.now().isoformat(),
            'total_segments': total_segments,
            'matched_segments': len(matching),
            'total_duration': total_duration,
            'matched_duration': match_duration,
        }
        
        # Generate outputs
        if args.output_text:
            save_text(matching, output_dir / f"{base_name}_{args.profile}.txt")
            save_text_with_timestamps(matching, output_dir / f"{base_name}_{args.profile}_timestamps.txt")
        
        if args.output_srt:
            save_srt(matching, output_dir / f"{base_name}_{args.profile}.srt")
        
        if args.output_json:
            save_json(matching, metadata, output_dir / f"{base_name}_{args.profile}.json")
        
        if args.output_audio:
            extract_audio_segments(
                audio_path,
                matching,
                str(output_dir / f"{base_name}_{args.profile}.wav")
            )
    
    print(f"\n✓ Extraction complete!")
    print(f"  Output directory: {output_dir}")
    
    # Show sample of what was extracted
    print(f"\n--- Sample of extracted text ---")
    for seg in matching[:5]:
        print(f"  [{seg['speaker_confidence']:.0%}] {seg['text'][:80]}...")
    if len(matching) > 5:
        print(f"  ... and {len(matching) - 5} more segments")


if __name__ == "__main__":
    main()
