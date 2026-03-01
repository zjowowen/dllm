from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.trainer import OneFlowTrainer


class OneFlowInterleavedTrainer(OneFlowTrainer):
    """
    Full interleaved-generation trainer (Algorithm 3).

    Reuses OneFlowTrainer with defaults for the full interleaved schedule:
    τ_text ~ Unif[0, 2], all samples have images, and the interleaved
    image time schedule (τ_img = τ_text - κ^{-1}(u)) is always active.

    This is the final training stage, combining:
    - Text insertion loss (CTMC or Paper Eq7)
    - Image flow matching loss (Eq 9)
    - Interleaved time schedule with image deletion/retention
    """

    @dataclass
    class OneFlowInterleavedConfig(OneFlowTrainer.OneFlowConfig):
        # Override defaults for full interleaved training.
        tau_text_max: float = 2.0
        image_loss_weight: float = 1.0
        text_loss_type: str = "ctmc"
        condition_text_on_time: bool = True
        normalize_text_loss_by_length: bool = True
        log_split_losses: bool = True
