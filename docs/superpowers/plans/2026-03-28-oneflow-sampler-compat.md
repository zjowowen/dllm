# OneFlow Sampler Compatibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py` with the renamed sampler base classes without changing OneFlow sampling behavior.

**Architecture:** This is a single-file API compatibility repair. Keep the existing local type aliases in `sampler.py`, but repoint them to `BaseSamplerConfig` and `BaseSamplerOutput` from the canonical base module so the file continues to expose the same local names while matching the current core sampler abstraction.

**Tech Stack:** Python import compatibility, `git diff`, targeted import/read verification.

---

## File Map

**Primary file**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`

**Reference file**
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/core/samplers/base.py`

**Out of scope**
- Do not modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md`
- Do not modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml`
- Do not modify: any other pipeline or test file

### Task 1: Verify the Base-Class Rename Mismatch

**Files:**
- Modify: none

- [ ] **Step 1: Prove the current base module exports the new names**

Run:

```bash
python - <<'PY'
from dllm.core.samplers.base import BaseSamplerConfig, BaseSamplerOutput
print(BaseSamplerConfig.__name__)
print(BaseSamplerOutput.__name__)
PY
```

Expected: prints `BaseSamplerConfig` and `BaseSamplerOutput`.

- [ ] **Step 2: Read the current `sampler.py` import line**

Read `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py:8` and confirm whether it still references old names or already contains the compatibility alias change.

- [ ] **Step 3: Capture the intended one-line repair**

Target import line:

```python
from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig as SamplerConfig, BaseSamplerOutput as SamplerOutput
```

Expected: this preserves the local `SamplerConfig` / `SamplerOutput` names while sourcing the new canonical base classes.

### Task 2: Apply the Single-File Compatibility Fix

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`

- [ ] **Step 1: Write the failing check in the smallest useful form**

Record the current mismatch as:

```text
`dllm/pipelines/oneflow/sampler.py` still imports old sampler base names, while `dllm/core/samplers/base.py` now exports `BaseSamplerConfig` and `BaseSamplerOutput`.
```

- [ ] **Step 2: Apply the one-line import fix**

Change:

```python
from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
```

to:

```python
from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig as SamplerConfig, BaseSamplerOutput as SamplerOutput
```

- [ ] **Step 3: Verify the file still uses the local aliases consistently**

Read the following lines after the edit and confirm no further code changes are needed:

```text
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py:17
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py:52
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py:77
```

Expected: `OneFlowSamplerConfig(SamplerConfig)`, `OneFlowSamplerOutput(SamplerOutput)`, and the return annotation still make sense without further edits.

### Task 3: Run Minimal Verification and Commit

**Files:**
- Modify only if needed: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`

- [ ] **Step 1: Run a targeted import verification**

Run:

```bash
python - <<'PY'
from dllm.pipelines.oneflow.sampler import OneFlowSamplerConfig, OneFlowSamplerOutput
print(OneFlowSamplerConfig.__mro__[1].__name__)
print(OneFlowSamplerOutput.__mro__[1].__name__)
PY
```

Expected: prints `BaseSamplerConfig` and `BaseSamplerOutput`.

- [ ] **Step 2: Review the final diff scope**

Run:

```bash
git diff -- /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py
```

Expected: exactly one import-line change, with no behavior changes elsewhere in the file.

- [ ] **Step 3: Commit**

```bash
git add /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/specs/2026-03-28-oneflow-sampler-compat-design.md /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/plans/2026-03-28-oneflow-sampler-compat.md
git commit -m "polish oneflow sampler compatibility"
```

Expected: one small compatibility-fix commit.
