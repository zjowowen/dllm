## OneFlow / EditFlow utils 复用与 SFT prompt_len 语义改造说明

本文档聚焦于工程层面的“复用/抽象”改造：把 `oneflow` 与 `editflow` 共同依赖的 CTMC 相关 helper 抽到共享模块，并补齐 OneFlow 的 `prompt_len`（SFT 不编辑 prompt）语义。

### 背景

`dllm` 同时维护了两条 pipeline：
- `dllm/pipelines/editflow`：文本 edit operations（SUB/DEL/INS）的 CTMC 训练与采样
- `dllm/pipelines/oneflow`：文本 insertion edit flow + 图像 latent flow matching 的多模态 interleaved 训练与采样

随着 OneFlow 引入自己的 trainer/sampler，出现了两类工程问题：
- **通用 helper 重复实现**：例如 padding、从 logits 采样、rate→Bernoulli 触发等。
- **pipeline 间相互 import**：OneFlow 训练侧曾直接 import `editflow.utils.pad_1d`，长期看会让依赖关系变得更脆弱。

### 改造目标

- 将 “CTMC 通用工具函数” 抽到共享模块，减少重复与漂移。
- 保持对外兼容：历史路径（如 `editflow.utils.pad_1d`）不应被破坏。
- 补齐 OneFlow 的 `prompt_len` 行为，使 SFT 能稳定保持 prefix conditioning。

### 共享模块边界

新增共享模块：
- `dllm/pipelines/ctmc_utils.py`

放入该模块的函数必须满足：
- **不依赖任何 pipeline**（避免循环 import）
- **尽量轻依赖**（当前仅依赖 `torch`）
- **语义明确且可复用**（EditFlow/OneFlow 都能用）

当前抽取内容：
- `pad_1d(batch_lists, pad_val) -> (out, mask)`
- `sample_from_logits(logits_row, temperature) -> int`
- `bernoulli_from_rate(rate, tau) -> Tensor`
- `safe_log(x, eps) -> Tensor`

### 代码迁移与兼容策略

迁移后的 import 关系：
- `dllm/pipelines/editflow/trainer.py`：从 `ctmc_utils` import `pad_1d`
- `dllm/pipelines/editflow/sampler.py`：从 `ctmc_utils` import `sample_from_logits` / `bernoulli_from_rate`
- `dllm/pipelines/oneflow/trainer.py`：从 `ctmc_utils` import `pad_1d`
- `dllm/pipelines/oneflow/sampler.py`：从 `ctmc_utils` import `sample_from_logits`

兼容策略：
- `dllm/pipelines/editflow/utils.py` 继续对外暴露 `pad_1d`（通过 `from dllm.pipelines.ctmc_utils import pad_1d` 的 re-export）。
  - 这样外部或历史代码若仍写 `from dllm.pipelines.editflow.utils import pad_1d` 不会破坏。

### OneFlow 的 prompt_len（SFT）语义

#### 定义

当数据为 `prompt + response`（SFT）格式时，`prompt_len` 表示 `x1_ids` 中 prompt 前缀长度（通常包含 BOS）。

#### 训练侧行为（OneFlowTrainer）

- 当 batch 提供 `prompt_len`：构造 `X_t` 时会强制 keep `x1_ids[:prompt_len]`，从而不在 prompt 内制造“删除→插入”的事件。
- 约束：
  - `0 < prompt_len <= len(x1_ids)`，否则报错
  - **多模态 v1 限制**：如果 batch 同时提供 `image_latents` 且 `prompt_len` 覆盖 `<|oneflow_image|>`，会显式报错
    - 原因：v1 尚未定义“prompt 内图像作为条件输入”的语义，避免把条件图像误当作待生成图像

#### 推断侧行为（OneFlowSampler）

推断侧已提供 `edit_prompt` 开关：
- `edit_prompt=False`（默认）：只允许从 prompt 最后一个 token 之后开始插入，保证 prefix conditioning 稳定。

### 风险与验证建议

#### 风险
- helper 抽取可能造成边界行为变化（例如 clamp/temperature 的处理差异）。
- prompt_len 的新约束可能让一些旧数据在 OneFlow 上报错（prompt 内含 `<|oneflow_image|>` 且传入 `image_latents`）。

#### 验证
- 为共享 helper 增加单元测试（形状、确定性、概率 clamp）。
- 用最小脚本 smoke：
  - `examples/editflow/*` 能 import 并运行到 forward
  - `examples/oneflow/*` 能 import 并运行到 forward

### 后续可扩展方向（v2 设想）

- 条件图像支持：区分「条件图像」与「待生成图像」的 token/字段（避免复用 `<|oneflow_image|>` 造成歧义）。
- 进一步归并数值稳定工具：例如统一使用 `ctmc_utils.safe_log`（当前 trainer 内也可继续局部定义，按需迁移）。


