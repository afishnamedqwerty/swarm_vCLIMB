from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
except ImportError:  # pragma: no cover - optional dependency
    data = None
    imagebind_model = None
    ModalityType = None


@dataclass
class ImageBindInputs:
    """Inputs for ImageBind across supported modalities."""

    vision: Optional[List[str]] = None
    audio: Optional[List[str]] = None
    text: Optional[List[str]] = None


class ImageBindEmbedder:
    """
    Thin wrapper around ImageBind for unified embeddings.

    Usage:
        embedder = ImageBindEmbedder()
        embeddings = embedder.embed(ImageBindInputs(vision=[...], audio=[...]))
    """

    def __init__(self, device: Optional[str] = None, pretrained: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model(pretrained)

    def _load_model(self, pretrained: bool):
        if imagebind_model is None or ModalityType is None:
            raise ImportError(
                "imagebind is required. Install from https://github.com/facebookresearch/ImageBind."
            )
        model = imagebind_model.imagebind_huge(pretrained=pretrained)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_inputs(self, inputs: ImageBindInputs):
        if data is None:
            raise ImportError(
                "imagebind data loaders not available. Install imagebind extras for data transforms."
            )
        payload = {}
        if inputs.vision:
            payload[ModalityType.VISION] = data.load_and_transform_vision_data(
                inputs.vision, self.device
            )
        if inputs.audio:
            payload[ModalityType.AUDIO] = data.load_and_transform_audio_data(
                inputs.audio, self.device
            )
        if inputs.text:
            payload[ModalityType.TEXT] = data.load_and_transform_text(
                inputs.text, self.device
            )
        if not payload:
            raise ValueError("At least one modality input is required for ImageBind.")
        return payload

    def embed(self, inputs: ImageBindInputs) -> Dict[str, np.ndarray]:
        """Embed provided modalities and return numpy arrays keyed by modality name."""
        payload = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model(payload)

        embeddings: Dict[str, np.ndarray] = {}
        for modality, tensor in outputs.items():
            embeddings[modality.name.lower()] = tensor.detach().cpu().numpy()
        return embeddings

    def embed_video(
        self,
        frame_paths: List[str],
        audio_path: Optional[str] = None,
        text: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Convenience wrapper for vision + optional audio/text."""
        inputs = ImageBindInputs(
            vision=frame_paths,
            audio=[audio_path] if audio_path else None,
            text=text,
        )
        return self.embed(inputs)
