"""Sampling and bootstrapping utilities for CLIMB mixtures."""

from .dirichlet_sampler import MixtureSampler
from .proxy import ProxyTrainer, VideoMixture
from .predictor import PerformancePredictor
from .loop import climb_bootstrap
from .proxy_impl import EmbeddingProxyTrainer, ProxyConfig, make_proxy_functions

__all__ = [
    "MixtureSampler",
    "ProxyTrainer",
    "EmbeddingProxyTrainer",
    "ProxyConfig",
    "make_proxy_functions",
    "PerformancePredictor",
    "VideoMixture",
    "climb_bootstrap",
]
