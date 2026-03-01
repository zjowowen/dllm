from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig


@dataclass
class OneFlowMixedGenerationSamplerConfig(OneFlowSamplerConfig):
    """
    Mixed-generation sampler config.

    Supports both text-only and text+image generation depending on
    whether the prompt contains `<|oneflow_image|>` tokens.
    """

    dt: float = 0.05
    max_steps: int = 40
    use_pi_gate: bool = True


class OneFlowMixedGenerationSampler(OneFlowSampler):
    """
    Mixed-generation sampler wrapper.

    Reuses OneFlowSampler for both text-only and text+image sampling.
    """

    pass
