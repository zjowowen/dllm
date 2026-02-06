# OneFlow 分支开发进度与下一步任务

本总结基于 `doc/oneflow` 现有文档与审计记录，聚焦“代码已实现/已对齐的能力”与“仍需验证/推进的工作”。如需具体执行步骤请回到对应文档。

## 当前进展（已在代码中落地或与论文对齐）
- **训练目标与采样逻辑已对齐论文算法**（Algorithm 1–3）：默认 `text_loss_type="paper"` + `condition_text_on_time=False`，覆盖 Eq(7) 文本 loss、interleaved schedule、image flow matching 等核心要点（详见代码对齐审计）。
- **混合生成采样策略可控**：支持 `mixed_generation_prob` 与 `τ_text` 分布切换。
- **文本/图像统一序列与时间调度**：具备 interleaved 时间表与图像插入/删除语义的实现约束。
- **数据准备与离线训练流程具备**：HF→img2dataset→latents→NPU 训练链路脚本已整理。
- **分阶段验证路线已定义**：Stage 0–5 归零式验证与 Debug Gate A/B/C 已形成可执行清单。

## 仍需验证或推进的里程碑
验收标准来自设计文档的里程碑定义：
- **text-only toy**：插入式生成可增长序列。
- **text+image toy**：训练后可采样出至少 1 张图像 latent，必要时可 VAE 解码。
- **工程可扩展性**：结构清晰、可扩展到多图/多模态/多分辨率。

## 下一步任务（建议执行顺序）
### 1) 归零式验证（确保实现正确）
- 按 Stage 0→5 依次跑通：数据/文本/图像/混合/交错/最小集成验证。
- 产物按 trace/json/png/metrics 统一输出，便于定位“哪条边开始偏离预期”。

### 2) 过拟合与排障闭环
- 依次通过 Gate A/B/C：数据可解码 → 训练确实优化 image loss → 用训练样本 reconstruction 指标随 step 改善。

### 3) 训练规模逐步放大（文本侧优先）
- 单机 16 卡小数据闭环 → 更大离线 PT 数据 → 2 节点 bring-up（200–500 steps） → 多机规模化（10B+ tokens） → SFT。

### 4) 工程改造与复用
- 抽取 CTMC 通用 helper 到共享模块并保持兼容。
- 完成 OneFlow `prompt_len` 语义与相关约束的实现/测试。

## 参考文档入口
- 里程碑定义：`doc/oneflow/design/oneflow_design_zh.md`
- 代码对齐审计：`doc/oneflow/validation/oneflow_paper_alignment_audit_2510_03506.md`
- 归零式验证：`doc/oneflow/validation/oneflow_zero_validation_zh.md`
- 过拟合排障：`doc/oneflow/validation/oneflow_overfit_debug_zh.md`
- 训练计划：`doc/oneflow/guides/oneflow_text_training_plan_npu_zh.md`

