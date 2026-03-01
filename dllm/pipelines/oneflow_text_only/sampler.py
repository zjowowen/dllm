from __future__ import annotations

from dataclasses import dataclass

from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig


@dataclass
class OneFlowTextOnlySamplerConfig(OneFlowSamplerConfig):
    """
    Text-only sampler config.

    v1 keeps parity with OneFlowSamplerConfig.
    """

    pass


class OneFlowTextOnlySampler(OneFlowSampler):
    """
    Text-only sampler wrapper.

    v1 intentionally reuses OneFlowSampler implementation and return format.
    """

    pass

