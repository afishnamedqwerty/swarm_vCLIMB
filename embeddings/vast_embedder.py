from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .imagebind_pipeline import ImageBindEmbedder
from .preprocess import AudioExtractor, FrameExtractor


TextEncoder = Callable[[List[str]], np.ndarray]
WhisperTranscriber = Callable[[str], Dict]


@dataclass
class VASTEmbedding:
    """Container for unified Video-Audio-Speech-Text embeddings."""

    visual: np.ndarray
    audio: Optional[np.ndarray]
    speech: Optional[np.ndarray]
    metadata: Optional[np.ndarray]
    fused: np.ndarray


class VASTEmbedder:
    """
    Build unified embeddings by combining ImageBind outputs with speech and metadata text.

    The embedder handles preprocessing (frame/audio extraction), optional ASR, and
    concatenation with metadata embeddings to match the CLIMB VAST recipe.
    """

    def __init__(
        self,
        imagebind: Optional[ImageBindEmbedder] = None,
        frame_extractor: Optional[FrameExtractor] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        whisper: Optional[WhisperTranscriber] = None,
        text_encoder: Optional[TextEncoder] = None,
        l2_normalize: bool = True,
    ):
        self.imagebind = imagebind or ImageBindEmbedder()
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.audio_extractor = audio_extractor or AudioExtractor()
        self.whisper = whisper
        self.text_encoder = text_encoder
        self.l2_normalize = l2_normalize

    @staticmethod
    def _l2_normalize(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None:
            return None
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    @staticmethod
    def _reduce_embedding(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=0)

    def _encode_transcript(self, audio_path: Optional[str]) -> Optional[np.ndarray]:
        if not audio_path or not self.whisper:
            return None
        transcript = self.whisper(audio_path)
        if isinstance(transcript, dict):
            text = transcript.get("text", "") or ""
        else:
            text = str(transcript)
        if not text or not self.text_encoder:
            return None
        encoded = self.text_encoder([text])
        return encoded[0] if encoded is not None else None

    def _encode_metadata(self, metadata_text: Optional[str]) -> Optional[np.ndarray]:
        if not metadata_text or not self.text_encoder:
            return None
        encoded = self.text_encoder([metadata_text])
        return encoded[0] if encoded is not None else None

    def embed_video(
        self,
        video_path: str,
        metadata_text: Optional[str] = None,
        precomputed_frames: Optional[List[str]] = None,
        precomputed_audio: Optional[str] = None,
    ) -> VASTEmbedding:
        """Generate a fused VAST embedding for a single video."""
        frame_paths = precomputed_frames or self.frame_extractor.extract(video_path)
        audio_path = precomputed_audio or self.audio_extractor.extract(video_path)

        imagebind_outputs = self.imagebind.embed_video(frame_paths, audio_path)

        visual = self._reduce_embedding(imagebind_outputs.get("vision"))
        audio = self._reduce_embedding(imagebind_outputs.get("audio"))
        speech = self._encode_transcript(audio_path)
        metadata = self._encode_metadata(metadata_text)

        components = []
        for comp in (visual, audio, speech, metadata):
            components.append(self._l2_normalize(comp) if self.l2_normalize else comp)

        fused_components = [c for c in components if c is not None]
        if not fused_components:
            raise ValueError("No embeddings produced for the provided video.")

        fused = np.concatenate(fused_components)
        if self.l2_normalize:
            fused = self._l2_normalize(fused)

        return VASTEmbedding(
            visual=visual,
            audio=audio,
            speech=speech,
            metadata=metadata,
            fused=fused,
        )
