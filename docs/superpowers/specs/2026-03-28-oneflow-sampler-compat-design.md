# OneFlow Sampler 兼容修补设计

## 背景

当前 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/core/samplers/base.py` 中已经明确使用新的基类命名：

- `BaseSamplerConfig`
- `BaseSamplerOutput`

但 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py` 原先仍沿用旧命名导入：

- `SamplerConfig`
- `SamplerOutput`

这导致 `OneFlowSamplerConfig` 与 `OneFlowSamplerOutput` 的继承基类名与当前核心抽象层不一致。主工作区里已有一处未提交改动，正是把该导入对齐到新的基类命名。

## 目标

- 让 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py` 与 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/core/samplers/base.py` 的真实导出名保持一致；
- 以最小改动完成兼容修补；
- 不扩大到无关文件或行为变更。

## 非目标

- 不修改 `OneFlowSampler` 的采样逻辑；
- 不重构类型命名风格；
- 不扫描和修复仓库其他文件中的类似问题；
- 不修改 README、依赖配置或测试策略。

## 推荐方案

采用最小兼容修补：

- 在 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py:8` 将旧命名导入改为：
  - `BaseSamplerConfig as SamplerConfig`
  - `BaseSamplerOutput as SamplerOutput`

这样可以保持文件内部已有类型名不变：

- `class OneFlowSamplerConfig(SamplerConfig)`
- `class OneFlowSamplerOutput(SamplerOutput)`
- `-> SamplerOutput | torch.Tensor`

同时避免对整份文件继续做重命名或签名清理。

## 为什么不选更大的方案

### 方案 A：最小兼容修补（推荐）

优点：

- 只改一行；
- 风险最低；
- 与当前主工作区已有未提交改动一致；
- 最适合作为单点兼容性提交。

### 方案 B：局部重命名清理

做法：

- 继续把文件内部的 `SamplerConfig` / `SamplerOutput` 直接改成 `BaseSamplerConfig` / `BaseSamplerOutput`

问题：

- 改动面更大；
- 对当前问题没有额外收益；
- 会把简单兼容修补扩展成风格调整。

### 方案 C：全仓库统一清理

做法：

- 搜索所有旧命名残留并统一替换

问题：

- 明显超出当前范围；
- 容易牵出无关改动；
- 不符合“只处理当前未提交改动”的目标。

## 文件范围

本次只涉及：

- 修改：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`
- 对照：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/core/samplers/base.py`

## 验收标准

完成后应满足：

1. `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py:8` 不再引用不存在的旧符号；
2. `OneFlowSamplerConfig` 与 `OneFlowSamplerOutput` 的继承关系仍然成立；
3. 文件其余逻辑不变；
4. 变更范围只限于这一个文件。

## 预期结果

完成后，这次改动将成为一个单点、低风险的 API 兼容修补，使 OneFlow sampler 继续跟随核心 sampler 抽象层的命名演进，而不引入额外行为变化。
