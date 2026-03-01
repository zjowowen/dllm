from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig


@dataclass
class OneFlowInterleavedSamplerConfig(OneFlowSamplerConfig):
    """
    Interleaved sampler config (Algorithm 1-2).

    Supports full interleaved generation: text Bernoulli insertion +
    image Euler integration, with dynamic image creation when
    <|oneflow_image|> tokens are sampled.
    """

    dt: float = 0.05
    max_steps: int = 40
    use_pi_gate: bool = True
    max_new_tokens: int = 256
    max_seq_len: int = 512
    condition_text_on_time: bool = True
    image_num_tokens: int = 256  # 16×16 latent grid


class OneFlowInterleavedSampler(OneFlowSampler):
    """
    Interleaved sampler (Algorithm 1-2).

    Reuses OneFlowSampler for coordinated text+image generation.
    Text tokens are inserted via Bernoulli sampling; image latents
    are denoised via Euler integration. New images are created when
    <|oneflow_image|> is sampled.
    """

    pass
