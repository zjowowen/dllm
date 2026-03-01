from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.trainer import OneFlowTrainer


class OneFlowImageOnlyTrainer(OneFlowTrainer):
    """
    Image-only pipeline trainer.

    Reuses the battle-tested OneFlowTrainer but with defaults that isolate
    image flow matching: τ_text is sampled in [tau_text_min, tau_text_max]
    where tau_text_min > 1 ensures text is fully preserved (κ(1)=1).

    The text loss (mostly π BCE pushing π→1 for empty bags) is still
    computed for gradient flow through the trunk, but the dominant
    optimization signal comes from the image velocity head.
    """

    @dataclass
    class OneFlowImageOnlyConfig(OneFlowTrainer.OneFlowConfig):
        # Override defaults for image-only isolation.
        tau_text_max: float = 2.0
        image_loss_weight: float = 1.0
        text_loss_type: str = "ctmc"
        condition_text_on_time: bool = False
        log_split_losses: bool = True
