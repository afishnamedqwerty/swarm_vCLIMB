"""Multimodal embedding utilities for the CLIMB pipeline."""

from .imagebind_pipeline import ImageBindEmbedder, ImageBindInputs
from .vast_embedder import VASTEmbedder, VASTEmbedding
from .preprocess import AudioExtractor, FrameExtractor

__all__ = [
    "AudioExtractor",
    "FrameExtractor",
    "ImageBindEmbedder",
    "ImageBindInputs",
    "VASTEmbedder",
    "VASTEmbedding",
]
