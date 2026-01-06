"""
OneFlow pipeline (multimodal): insertion-based text edit flow + image-latent flow matching.

This package is intentionally lightweight at import time. Heavy third-party deps
used by the Transfusion backbone are only imported inside the modules that need them.
"""

from . import trainer, sampler, utils
from .trainer import OneFlowTrainer
from .sampler import OneFlowSampler, OneFlowSamplerConfig


