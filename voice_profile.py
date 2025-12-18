#!/usr/bin/env python3
"""
Voice Profile Manager CLI
Create and manage voice profiles for speaker identification.

Usage:
    # Create a profile from a video (uses first 60 seconds)
    python voice_profile.py create kyle ~/Videos/me_talking.mp4
    
    # Create from specific time range (you speaking from 0:30 to 1:30)
    python voice_profile.py create kyle video.mp4 --start 30 --end 90
    
    # Create from audio file
    python voice_profile.py create kyle my_voice.wav
    
    # List all profiles
    python voice_profile.py list
    
    # Test a profile against a file
    python voice_profile.py test kyle unknown_speaker.wav
    
    # Delete a profile
    python voice_profile.py delete kyle
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed."""
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
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print("Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def cmd_create(args):
    """Create a new voice profile."""
    from speaker_id import SpeakerIdentifier
    
    identifier = SpeakerIdentifier(args.profiles_dir)
    
    source = Path(args.source)
    if not source.exists():
        print(f"Error: File not found: {source}")
        sys.exit(1)
    
    # Check if it's video or audio
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    
    ext = source.suffix.lower()
    
    try:
        if ext in video_exts:
            print(f"Creating voice profile '{args.name}' from video: {source.name}")
            if args.start is not None or args.end is not None:
                print(f"  Using time range: {args.start or 0}s - {args.end or 'end'}s")
            
            profile = identifier.create_profile_from_video(
                args.name,
                str(source),
                start=args.start,
                end=args.end,
                overwrite=args.overwrite
            )
        elif ext in audio_exts:
            print(f"Creating voice profile '{args.name}' from audio: {source.name}")
            profile = identifier.create_profile(
                args.name,
                [str(source)],
                overwrite=args.overwrite
            )
        else:
            print(f"Error: Unsupported file type: {ext}")
            print(f"Supported: {video_exts | audio_exts}")
            sys.exit(1)
        
        print(f"\n✓ Voice profile '{args.name}' created successfully!")
        print(f"  Saved to: {args.profiles_dir}/profiles.json")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_list(args):
    """List all voice profiles."""
    from speaker_id import SpeakerIdentifier
    
    identifier = SpeakerIdentifier(args.profiles_dir)
    profiles = identifier.list_profiles()
    
    if not profiles:
        print("No voice profiles found.")
        print(f"\nCreate one with:")
        print(f"  python voice_profile.py create <name> <video_or_audio_file>")
        return
    
    print(f"Voice Profiles ({len(profiles)}):")
    print("-" * 40)
    
    for name in profiles:
        profile = identifier.get_profile(name)
        print(f"  • {name}")
        print(f"    Created: {profile.created_at}")
        if profile.sample_files:
            print(f"    Samples: {len(profile.sample_files)} file(s)")
    
    print("-" * 40)


def cmd_delete(args):
    """Delete a voice profile."""
    from speaker_id import SpeakerIdentifier
    
    identifier = SpeakerIdentifier(args.profiles_dir)
    
    if args.name not in identifier.list_profiles():
        print(f"Error: Profile '{args.name}' not found.")
        sys.exit(1)
    
    if not args.yes:
        confirm = input(f"Delete profile '{args.name}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    
    identifier.delete_profile(args.name)
    print(f"✓ Profile '{args.name}' deleted.")


def cmd_test(args):
    """Test a profile against an audio/video file."""
    from speaker_id import SpeakerIdentifier
    
    identifier = SpeakerIdentifier(args.profiles_dir)
    
    if args.profile not in identifier.list_profiles():
        print(f"Error: Profile '{args.profile}' not found.")
        print(f"Available profiles: {identifier.list_profiles()}")
        sys.exit(1)
    
    source = Path(args.source)
    if not source.exists():
        print(f"Error: File not found: {source}")
        sys.exit(1)
    
    print(f"Testing '{args.profile}' against: {source.name}")
    print("-" * 40)
    
    # Extract audio if video
    import subprocess
    import tempfile
    
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    
    if source.suffix.lower() in video_exts:
        print("Extracting audio from video...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = [
            'ffmpeg', '-y', '-i', str(source),
            '-vn', '-ar', '16000', '-ac', '1',
            '-loglevel', 'error',
            tmp_path
        ]
        subprocess.run(cmd, check=True)
        audio_path = tmp_path
    else:
        audio_path = str(source)
    
    try:
        result = identifier.identify_speaker(audio_path, threshold=0.0)
        
        if result:
            match_name, confidence = result
            
            print(f"\nResults:")
            print(f"  Best match: {match_name}")
            print(f"  Confidence: {confidence:.1%}")
            
            if match_name == args.profile:
                if confidence >= 0.7:
                    print(f"\n✓ HIGH MATCH - This sounds like '{args.profile}'!")
                elif confidence >= 0.5:
                    print(f"\n~ POSSIBLE MATCH - Might be '{args.profile}'")
                else:
                    print(f"\n✗ LOW MATCH - Probably not '{args.profile}'")
            else:
                print(f"\n✗ Best match is '{match_name}', not '{args.profile}'")
        else:
            print("\nNo profiles matched.")
            
    finally:
        if source.suffix.lower() in video_exts:
            import os
            os.unlink(tmp_path)


def cmd_info(args):
    """Show detailed info about a profile."""
    from speaker_id import SpeakerIdentifier
    
    identifier = SpeakerIdentifier(args.profiles_dir)
    
    profile = identifier.get_profile(args.name)
    if not profile:
        print(f"Error: Profile '{args.name}' not found.")
        sys.exit(1)
    
    print(f"Voice Profile: {profile.name}")
    print("=" * 40)
    print(f"Created:    {profile.created_at}")
    print(f"Embedding:  {len(profile.embedding)} dimensions")
    print(f"Sample files:")
    for f in profile.sample_files:
        print(f"  • {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage voice profiles for speaker identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a profile from a video where you're speaking
  python voice_profile.py create kyle ~/Videos/me_presenting.mp4
  
  # Create from a specific segment (you at 0:30-1:30)
  python voice_profile.py create kyle video.mp4 --start 30 --end 90
  
  # Test if a file matches your voice
  python voice_profile.py test kyle unknown_video.mp4
  
  # List all profiles
  python voice_profile.py list
"""
    )
    
    parser.add_argument('--profiles-dir', default='voice_profiles',
                        help='Directory to store voice profiles')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new voice profile')
    create_parser.add_argument('name', help='Name for the profile (e.g., "kyle")')
    create_parser.add_argument('source', help='Video or audio file with your voice')
    create_parser.add_argument('--start', type=float, help='Start time in seconds')
    create_parser.add_argument('--end', type=float, help='End time in seconds')
    create_parser.add_argument('--overwrite', action='store_true',
                               help='Overwrite existing profile')
    create_parser.set_defaults(func=cmd_create)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all voice profiles')
    list_parser.set_defaults(func=cmd_list)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a voice profile')
    delete_parser.add_argument('name', help='Profile name to delete')
    delete_parser.add_argument('-y', '--yes', action='store_true',
                               help='Skip confirmation')
    delete_parser.set_defaults(func=cmd_delete)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a profile against a file')
    test_parser.add_argument('profile', help='Profile name to test')
    test_parser.add_argument('source', help='Audio or video file to test')
    test_parser.set_defaults(func=cmd_test)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show profile details')
    info_parser.add_argument('name', help='Profile name')
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    check_dependencies()
    args.func(args)


if __name__ == "__main__":
    main()
