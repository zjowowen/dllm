from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.trainer import OneFlowTrainer


class OneFlowMixedGenerationTrainer(OneFlowTrainer):
    """
    Mixed-generation pipeline trainer.

    Reuses OneFlowTrainer with defaults oriented toward joint text+image
    training at a controllable mixing ratio (`mixed_generation_prob`).

    Key difference from full interleaved: `mixed_generation_prob < 1.0`
    means some samples are text-only (no image schedule), providing a
    controlled environment to verify text/image loss compatibility before
    moving to full interleaved.
    """

    @dataclass
    class OneFlowMixedGenerationConfig(OneFlowTrainer.OneFlowConfig):
        # Override defaults for mixed-generation experiments.
        tau_text_max: float = 2.0
        image_loss_weight: float = 1.0
        text_loss_type: str = "ctmc"
        condition_text_on_time: bool = True
        normalize_text_loss_by_length: bool = True
        log_split_losses: bool = True
