from __future__ import annotations

from dllm.pipelines.oneflow.trainer import OneFlowTrainer


class OneFlowTextOnlyTrainer(OneFlowTrainer):
    """
    Text-only pipeline trainer.

    v1 intentionally reuses the battle-tested OneFlowTrainer implementation
    (Eq(7) insertion loss path) to isolate model-backbone changes.
    """

    pass

