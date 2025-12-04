from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .proxy import ProxyTrainer


class EmbeddingDataset(Dataset):
    """Simple dataset wrapping cached embeddings and labels."""

    def __init__(self, rows: List[Tuple[np.ndarray, int]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        emb, label = self.rows[idx]
        return torch.from_numpy(emb).float(), torch.tensor(label, dtype=torch.long)


@dataclass
class ProxyConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 256
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"


class EmbeddingProxyTrainer(ProxyTrainer):
    """
    ProxyTrainer that builds dataloaders from cached embeddings + labels.

    Assumes embeddings are precomputed (e.g., VAST vectors) and stored in memory
    so proxy training is inexpensive.
    """

    def __init__(
        self,
        videos_by_cluster: Dict[int, Sequence[str]],
        features: Dict[str, np.ndarray],
        labels: Dict[str, int],
        cfg: ProxyConfig,
        tokens_budget: int = 1_000_000_000,
    ):
        super().__init__(videos_by_cluster, tokens_budget=tokens_budget)
        self.features = features
        self.labels = labels
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    def build_dataloader(self, videos: Iterable[str]) -> DataLoader:
        rows: List[Tuple[np.ndarray, int]] = []
        for path in videos:
            if path not in self.features or path not in self.labels:
                continue
            rows.append((self.features[path], int(self.labels[path])))
        dataset = EmbeddingDataset(rows)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

    def build_calibration_loader(self) -> DataLoader:
        rows: List[Tuple[np.ndarray, int]] = []
        for path, label in self.labels.items():
            if path in self.features:
                rows.append((self.features[path], int(label)))
        dataset = EmbeddingDataset(rows)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False)


def make_proxy_functions(trainer: EmbeddingProxyTrainer):
    """
    Create (build_model, train_fn, eval_fn) closures for use with climb_bootstrap.
    """
    cfg = trainer.cfg

    def build_model():
        model = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )
        return model.to(trainer.device)

    def train_fn(model, dataloader):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        for _ in range(cfg.epochs):
            for features, labels in dataloader:
                features = features.to(trainer.device)
                labels = labels.to(trainer.device)
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

    def eval_fn(model) -> float:
        model.eval()
        loader = trainer.build_calibration_loader()
        if len(loader.dataset) == 0:
            return 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(trainer.device)
                labels = labels.to(trainer.device)
                logits = model(features)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return correct / max(total, 1)

    return build_model, train_fn, eval_fn
