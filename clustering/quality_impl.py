import json
import subprocess
from typing import Dict, Mapping

from .quality import QualityScorer


class FFprobeQualityScorer(QualityScorer):
    """
    Lightweight quality scorer using ffprobe metadata.

    Heuristics (0-5):
    - visual_quality: resolution + video bitrate
    - audio_quality: presence + audio bitrate
    - educational_value / entertainment_value: neutral priors (3.0) to be refined later
    """

    def __init__(self, min_width: int = 640, min_bitrate_kbps: int = 500):
        self.min_width = min_width
        self.min_bitrate_kbps = min_bitrate_kbps

    def _ffprobe(self, video_path: str) -> Dict:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            video_path,
        ]
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
        return json.loads(out.decode("utf-8"))

    def _visual_score(self, streams: list) -> float:
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            return 0.0
        vs = video_streams[0]
        width = int(vs.get("width", 0) or 0)
        height = int(vs.get("height", 0) or 0)
        bitrate = float(vs.get("bit_rate", 0) or 0) / 1000.0

        res_score = min(5.0, max(0.0, width / self.min_width))
        bitrate_score = min(5.0, max(0.0, bitrate / self.min_bitrate_kbps))
        return float((res_score + bitrate_score) / 2.0)

    def _audio_score(self, streams: list) -> float:
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if not audio_streams:
            return 0.0
        a = audio_streams[0]
        bitrate = float(a.get("bit_rate", 0) or 0) / 1000.0
        channels = int(a.get("channels", 0) or 0)
        sr = int(a.get("sample_rate", 0) or 0)

        bitrate_score = min(5.0, max(0.0, bitrate / 128.0))
        channel_score = 5.0 if channels >= 2 else 3.0
        sr_score = 5.0 if sr >= 44100 else 3.0
        return float((bitrate_score + channel_score + sr_score) / 3.0)

    def score(self, video_path: str) -> Mapping[str, float]:
        meta = self._ffprobe(video_path)
        streams = meta.get("streams", [])

        visual = self._visual_score(streams)
        audio = self._audio_score(streams)

        return {
            "visual_quality": visual,
            "audio_quality": audio,
            "educational_value": 3.0,
            "entertainment_value": 3.0,
            "content_safety": 3.0,
        }
