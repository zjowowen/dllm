from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig


@dataclass
class OneFlowImageOnlySamplerConfig(OneFlowSamplerConfig):
    """
    Image-only sampler config.

    Defaults tuned for image-conditioned generation (text prompt is provided,
    model generates image latents via Euler integration).
    """

    dt: float = 0.05
    max_steps: int = 40
    use_pi_gate: bool = True
    image_num_tokens: int = 256  # 16×16 latent grid


class OneFlowImageOnlySampler(OneFlowSampler):
    """
    Image-only sampler wrapper.

    Reuses OneFlowSampler; intended for image-conditioned generation
    where the text prompt is fixed and only image latents are denoised.
    """

    pass
