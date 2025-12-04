import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class FrameExtractor:
    """Extract frames from video files using ffmpeg."""

    fps: int = 1
    output_dir: Optional[Path] = None

    def extract(self, video_path: str) -> List[str]:
        """
        Extract frames at a fixed FPS.

        Returns a list of frame file paths suitable for ImageBind vision input.
        """
        video_path = os.fspath(video_path)
        target_dir = Path(self.output_dir) if self.output_dir else Path(
            tempfile.mkdtemp(prefix="frames_")
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        frame_pattern = str(target_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={self.fps}",
            "-q:v",
            "2",
            frame_pattern,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        frames = sorted(str(p) for p in target_dir.glob("frame_*.jpg"))
        return frames


@dataclass
class AudioExtractor:
    """Extract audio tracks from video files."""

    sample_rate: int = 16000
    output_dir: Optional[Path] = None

    def extract(self, video_path: str) -> str:
        """
        Extract audio as mono WAV for downstream audio and ASR models.

        Returns the path to the extracted audio file.
        """
        video_path = os.fspath(video_path)
        target_dir = Path(self.output_dir) if self.output_dir else Path(
            tempfile.mkdtemp(prefix="audio_")
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        audio_path = target_dir / "audio.wav"
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return str(audio_path)
