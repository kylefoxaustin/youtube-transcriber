#!/usr/bin/env python3
"""
Speaker Identification Module
Create voice profiles and identify/extract specific speakers from audio.

Uses SpeechBrain's speaker verification model to create voice embeddings
and match speakers across different recordings.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load heavy dependencies
_speaker_model = None
_speechbrain_available = None


def check_speechbrain_available() -> bool:
    """Check if SpeechBrain is installed."""
    global _speechbrain_available
    if _speechbrain_available is None:
        try:
            import speechbrain
            import torchaudio
            _speechbrain_available = True
        except ImportError:
            _speechbrain_available = False
    return _speechbrain_available


def get_speaker_model():
    """Lazy load the speaker verification model."""
    global _speaker_model
    
    if not check_speechbrain_available():
        raise ImportError(
            "SpeechBrain not installed. Install with:\n"
            "  pip install speechbrain torchaudio"
        )
    
    if _speaker_model is None:
        from speechbrain.inference.speaker import SpeakerRecognition
        
        logger.info("Loading speaker verification model...")
        _speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speaker_recognition"
        )
        logger.info("Speaker model loaded successfully")
    
    return _speaker_model


class VoiceProfile:
    """Represents a voice profile with embedding and metadata."""
    
    def __init__(self, name: str, embedding: np.ndarray, 
                 sample_files: List[str] = None, created_at: str = None):
        self.name = name
        self.embedding = embedding
        self.sample_files = sample_files or []
        self.created_at = created_at or self._get_timestamp()
    
    @staticmethod
    def _get_timestamp() -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'embedding': self.embedding.tolist(),
            'sample_files': self.sample_files,
            'created_at': self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VoiceProfile':
        return cls(
            name=data['name'],
            embedding=np.array(data['embedding']),
            sample_files=data.get('sample_files', []),
            created_at=data.get('created_at'),
        )


class SpeakerIdentifier:
    """
    Identifies and extracts specific speakers from audio.
    
    Usage:
        identifier = SpeakerIdentifier()
        
        # Create a profile from audio samples
        identifier.create_profile("kyle", ["sample1.wav", "sample2.wav"])
        
        # Or from a video (extracts audio automatically)
        identifier.create_profile_from_video("kyle", "me_speaking.mp4")
        
        # Find matching segments in another file
        matches = identifier.find_speaker_segments("kyle", "family_video.mp4", segments)
    """
    
    def __init__(self, profiles_dir: str = "voice_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, VoiceProfile] = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load all saved profiles from disk."""
        profiles_file = self.profiles_dir / "profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                for name, profile_data in data.items():
                    self.profiles[name] = VoiceProfile.from_dict(profile_data)
                logger.info(f"Loaded {len(self.profiles)} voice profiles")
            except Exception as e:
                logger.error(f"Failed to load profiles: {e}")
    
    def _save_profiles(self):
        """Save all profiles to disk."""
        profiles_file = self.profiles_dir / "profiles.json"
        data = {name: profile.to_dict() for name, profile in self.profiles.items()}
        with open(profiles_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def list_profiles(self) -> List[str]:
        """List all available voice profiles."""
        return list(self.profiles.keys())
    
    def get_profile(self, name: str) -> Optional[VoiceProfile]:
        """Get a profile by name."""
        return self.profiles.get(name)
    
    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        if name in self.profiles:
            del self.profiles[name]
            self._save_profiles()
            logger.info(f"Deleted profile: {name}")
            return True
        return False
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from an audio file."""
        model = get_speaker_model()
        embedding = model.encode_batch(
            model.load_audio(audio_path).unsqueeze(0)
        )
        return embedding.squeeze().cpu().numpy()
    
    def extract_embedding_from_segment(self, audio_path: str, 
                                        start: float, end: float) -> np.ndarray:
        """Extract speaker embedding from a specific segment of audio."""
        # Extract segment using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start), '-t', str(end - start),
                '-ar', '16000', '-ac', '1',
                '-loglevel', 'error',
                tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return self.extract_embedding(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def create_profile(self, name: str, audio_files: List[str], 
                       overwrite: bool = False) -> VoiceProfile:
        """
        Create a voice profile from one or more audio samples.
        
        For best results, use 30-60 seconds of clear speech.
        Multiple samples will be averaged for a more robust profile.
        """
        if name in self.profiles and not overwrite:
            raise ValueError(f"Profile '{name}' already exists. Use overwrite=True to replace.")
        
        if not audio_files:
            raise ValueError("At least one audio file is required")
        
        logger.info(f"Creating voice profile '{name}' from {len(audio_files)} sample(s)")
        
        embeddings = []
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                logger.warning(f"File not found: {audio_file}")
                continue
            
            try:
                emb = self.extract_embedding(audio_file)
                embeddings.append(emb)
                logger.info(f"  Processed: {audio_file}")
            except Exception as e:
                logger.error(f"  Failed to process {audio_file}: {e}")
        
        if not embeddings:
            raise ValueError("No valid audio files could be processed")
        
        # Average embeddings for a more robust profile
        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        profile = VoiceProfile(
            name=name,
            embedding=avg_embedding,
            sample_files=[str(f) for f in audio_files],
        )
        
        self.profiles[name] = profile
        self._save_profiles()
        
        logger.info(f"Voice profile '{name}' created successfully")
        return profile
    
    def create_profile_from_video(self, name: str, video_path: str,
                                   start: float = None, end: float = None,
                                   overwrite: bool = False) -> VoiceProfile:
        """
        Create a voice profile from a video file.
        
        Optionally specify start/end times to use only a specific segment.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Extract audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            cmd = ['ffmpeg', '-y', '-i', str(video_path)]
            if start is not None:
                cmd.extend(['-ss', str(start)])
            if end is not None and start is not None:
                cmd.extend(['-t', str(end - start)])
            elif end is not None:
                cmd.extend(['-t', str(end)])
            cmd.extend(['-vn', '-ar', '16000', '-ac', '1', '-loglevel', 'error', tmp_path])
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            return self.create_profile(name, [tmp_path], overwrite=overwrite)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two embeddings and return similarity score.
        
        Returns:
            Cosine similarity score between 0 and 1.
            Higher = more similar. Typically >0.7 indicates same speaker.
        """
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(np.clip(similarity, 0, 1))
    
    def identify_speaker(self, audio_path: str, threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Identify which profile (if any) matches the speaker in the audio.
        
        Returns:
            Tuple of (profile_name, confidence) or None if no match above threshold.
        """
        if not self.profiles:
            logger.warning("No voice profiles available")
            return None
        
        embedding = self.extract_embedding(audio_path)
        
        best_match = None
        best_score = 0.0
        
        for name, profile in self.profiles.items():
            score = self.compare_embeddings(embedding, profile.embedding)
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= threshold:
            return (best_match, best_score)
        return None
    
    def find_speaker_segments(self, profile_name: str, audio_path: str,
                               segments: List[dict], threshold: float = 0.6,
                               min_duration: float = 1.0) -> List[dict]:
        """
        Find segments in audio that match a voice profile.
        
        Args:
            profile_name: Name of the voice profile to match
            audio_path: Path to audio file
            segments: List of segments with 'start', 'end', 'text' keys
                      (typically from transcription)
            threshold: Minimum similarity score to consider a match (0-1)
            min_duration: Minimum segment duration to analyze (seconds)
        
        Returns:
            List of matching segments with added 'speaker_match' and 'confidence' keys
        """
        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        matching_segments = []
        total = len(segments)
        
        for i, segment in enumerate(segments):
            duration = segment['end'] - segment['start']
            
            # Skip very short segments
            if duration < min_duration:
                continue
            
            try:
                # Extract embedding for this segment
                seg_embedding = self.extract_embedding_from_segment(
                    audio_path, segment['start'], segment['end']
                )
                
                # Compare with profile
                similarity = self.compare_embeddings(seg_embedding, profile.embedding)
                
                segment_copy = segment.copy()
                segment_copy['speaker_confidence'] = similarity
                segment_copy['is_target_speaker'] = similarity >= threshold
                
                if similarity >= threshold:
                    segment_copy['speaker_match'] = profile_name
                    matching_segments.append(segment_copy)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Analyzed {i + 1}/{total} segments...")
                    
            except Exception as e:
                logger.warning(f"Failed to process segment {segment['start']:.1f}-{segment['end']:.1f}: {e}")
        
        logger.info(f"Found {len(matching_segments)}/{total} segments matching '{profile_name}' "
                   f"(threshold={threshold})")
        
        return matching_segments
    
    def label_transcript_segments(self, profile_name: str, audio_path: str,
                                   transcript: dict, threshold: float = 0.6) -> dict:
        """
        Add speaker identification labels to all transcript segments.
        
        For each segment, adds:
        - 'is_target_speaker': True/False/None
        - 'speaker_confidence': similarity score (0-1)
        
        Returns modified transcript dict.
        """
        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        transcript = transcript.copy()
        segments = transcript.get('segments', [])
        total = len(segments)
        target_count = 0
        
        logger.info(f"Labeling {total} segments for speaker '{profile_name}'...")
        
        for i, segment in enumerate(segments):
            duration = segment['end'] - segment['start']
            
            if duration < 0.5:  # Skip very short
                segment['is_target_speaker'] = None
                segment['speaker_confidence'] = 0.0
                continue
            
            try:
                seg_embedding = self.extract_embedding_from_segment(
                    audio_path, segment['start'], segment['end']
                )
                similarity = self.compare_embeddings(seg_embedding, profile.embedding)
                
                segment['is_target_speaker'] = similarity >= threshold
                segment['speaker_confidence'] = round(similarity, 3)
                
                if segment['is_target_speaker']:
                    target_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to analyze segment: {e}")
                segment['is_target_speaker'] = None
                segment['speaker_confidence'] = 0.0
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{total} segments...")
        
        transcript['segments'] = segments
        transcript['speaker_profile'] = profile_name
        transcript['speaker_threshold'] = threshold
        transcript['target_speaker_segments'] = target_count
        
        logger.info(f"Labeled transcript: {target_count}/{total} segments match '{profile_name}'")
        
        return transcript
    
    def extract_speaker_text(self, profile_name: str, audio_path: str,
                              segments: List[dict], threshold: float = 0.6) -> str:
        """
        Extract only the text from segments matching a voice profile.
        
        Returns concatenated text from all matching segments.
        """
        matching = self.find_speaker_segments(profile_name, audio_path, segments, threshold)
        
        texts = [seg['text'] for seg in matching if seg.get('text')]
        return ' '.join(texts)


def demo():
    """Demo usage of speaker identification."""
    print("=" * 50)
    print("Speaker Identification Module")
    print("=" * 50)
    
    if not check_speechbrain_available():
        print("\n⚠️  SpeechBrain not installed. Install with:")
        print("   pip install speechbrain torchaudio")
        return
    
    identifier = SpeakerIdentifier()
    profiles = identifier.list_profiles()
    
    print(f"\n✓ SpeechBrain available")
    print(f"✓ Loaded {len(profiles)} voice profile(s)")
    
    if profiles:
        print(f"  Profiles: {', '.join(profiles)}")
    
    print("\n" + "-" * 50)
    print("Usage Examples:")
    print("-" * 50)
    print("""
# Create a voice profile from a video where only you speak
identifier.create_profile_from_video("kyle", "just_me_talking.mp4")

# Or from an audio file
identifier.create_profile("kyle", ["my_voice_sample.wav"])

# Find your segments in another video
matching = identifier.find_speaker_segments("kyle", "family_video.mp4", transcript['segments'])

# Label all segments in a transcript
labeled = identifier.label_transcript_segments("kyle", "video.wav", transcript)

# Extract just your text
my_text = identifier.extract_speaker_text("kyle", "video.wav", segments)
""")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
