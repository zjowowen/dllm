# OneFlow 分支开发进度与下一步任务

本文是 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow` 的**当前权威状态页**：只记录“已实现 / 已验证 / 待完成或待澄清”的事实，不把设计目标当作已完成结论。具体执行步骤请回到对应专题文档。

## 当前状态总览

### 已实现
- **训练目标与主干采样逻辑已基本按论文落地**：默认 `text_loss_type="paper"` + `condition_text_on_time=False`，覆盖 Eq(7) 文本 loss、interleaved schedule、image flow matching 等主干要点（详见对齐审计）。
- **文本 / 图像统一序列与时间调度已具备代码骨架**：包含 interleaved 时间表，以及图像插入 / 删除相关语义约束。
- **image-only / mixed / interleaved 目前主要只有 wrapper 与入口**：训练、采样、验证路线已有文档和脚本入口，但多数仍停留在“可进入流程”，不等于功能链路已充分验证。
- **数据准备与离线训练流程已具备**：HF → img2dataset → latents → NPU 训练链路脚本已整理。
- **分阶段验证路线已定义**：Stage 0–5 归零式验证与 Debug Gate A/B/C 已形成清单。

### 已验证
- **text-only 是当前最完整、最可信的验证主线**：已有训练、评测、checkpoint sweep、EditFlow 对齐实验与 H200 离线执行记录支撑。
- **EditFlow 对齐实验已验证文本侧若干关键结论**：时间条件化单独收益有限，数据量与 CTMC 向量化更关键。
- **H200 离线 PT + 伪 SFT 框架已完成文本侧工具化补齐**：包含本地数据校验、启动脚本、伪 SFT 构建、批量 prompt 评测与 loss 聚合。

### 待完成或待澄清
- **image-only / mixed / interleaved 仍以 wrapper 与入口为主**：还不能把这些路径表述为“已稳定跑通”或“已充分验收”。
- **`mixed_generation_prob` 不能再表述为“已 fully controllable 的核心 mixed-generation 开关”**：当前只能确认存在相关参数与分布分支，仍需补足核心路径语义、实际生效点与验证结论。
- **sampler v1 仍受限于 `bs=1`，且 infill 尚未实现**：这两点继续约束图像 / 混合 / 交错采样结论的可信度。
- **interleaved 闭环仍未完成**：需要把数据、训练、采样、可视化与验收指标真正串成闭环。

## H200 离线 PT+伪 SFT 框架（2026-02-11）
- 已新增 fineweb 本地路径只读校验脚本：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/check_fineweb_local_readiness.py`
- 已新增 H200 PT 启动脚本（离线 + wandb offline + tensorboard）：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/launch_pt_text_h200.sh`
- 已新增 PT→伪 SFT 数据构建脚本：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/build_pseudo_sft_from_pt.py`
- 已新增 H200 伪 SFT 启动脚本：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/launch_sft_text_h200.sh`
- 已新增批量 prompt 评测脚本与模板集：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_prompts.py`、`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_text_minimal.jsonl`
- 已补齐文本侧 H200 启动脚本的显式模型参数入口：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/launch_pt_text_h200.sh` 与 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/launch_sft_text_h200.sh` 可直接接收并透传 `--dim/--depth/--heads/--dim_head/--dim_latent`；仓库中当前未提供 `MODEL_SIZE_PRESET` 包装脚本或 `0p6b/0p9b/1p1b/1p3b/custom` 这类预设切换层。
- 已增强稳定评测能力：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_loss.py` 支持 `--seeds` 多 seed 聚合与 `--output_json`
- 已新增 PT checkpoint 自动选点脚本：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_checkpoint_sweep.py`（支持 `hybrid/loss_tok/loss_total/prompt_first` 排序）
- 已新增 H200 执行手册与实时进展日志：
  - `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/oneflow_h200_offline_pt_pseudo_sft_zh.md`
  - `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/oneflow_h200_progress_log.md`

## 仍需验证或推进的里程碑
验收标准来自设计文档的里程碑定义：
- **text-only toy**：插入式生成可增长序列。
- **text+image toy**：训练后可采样出至少 1 张图像 latent，必要时可 VAE 解码。
- **工程可扩展性**：结构清晰、可扩展到多图/多模态/多分辨率。

## 下一步任务（建议执行顺序）
### 1) image-only 验证优先
- 先把 image-only toy 路线真正跑通并留痕：数据可解码、训练确实优化 image loss、采样可产出至少 1 张可检查的图像 latent / VAE 解码结果。
- 产物按 trace / json / png / metrics 统一输出，先确认“纯图路径是否闭环成立”。

### 2) mixed-generation 控制语义澄清
- 逐项厘清 `mixed_generation_prob`、`τ_text` 与相关入口在核心采样路径中的实际作用、生效条件与限制。
- 在验证完成前，不再把 mixed-generation 描述为“已可稳定 controllable 的核心能力”。

### 3) interleaved 闭环收口
- 在 image-only 与 mixed 控制语义清楚后，再补齐 interleaved 的数据、训练、采样、可视化与验收标准。

### 4) 训练规模逐步放大（文本侧延续）
- 继续沿 text-only 已验证路线推进：单机 16 卡小数据闭环 → 更大离线 PT 数据 → 2 节点 bring-up（200–500 steps） → 多机规模化（10B+ tokens） → SFT。

### 5) 工程改造与复用
- 抽取 CTMC 通用 helper 到共享模块并保持兼容。
- 完成 OneFlow `prompt_len` 语义与相关约束的实现 / 测试。

### 6) EditFlow 对齐实验（面向 text-only 收敛优化）
基于 EditFlow 与 OneFlow 的对比分析（详见 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/oneflow_editflow_textonly_comparison_zh.md`），按重要程度依次验证以下改进：

| 实验 | 变更项 | 状态 | 结论 |
|------|--------|------|------|
| **A. 时间条件化** | `--condition_text_on_time` | ✅ 已验证 | loss ~29-34 波动，比基线高；时间条件化单独不足以改善 |
| **B. CTMC loss + w(t)** | `--text_loss_type ctmc` | ✅ 已解除 | Python 循环已向量化 (~50x 加速)，详见 B2 |
| **C. 权重共享** | `--tie_q_logits_to_embedding` | ✅ 已验证 | 初始 loss 高但快速下降，稳态 ~35-42 |
| **D. 组合 A+C** | 时间条件化 + 权重共享 | ✅ 已验证 | 与 C 几乎一致，小数据上时间条件化无附加效果 |
| **E. 大数据集** | D + fineweb-edu 100k | ✅ 已验证 | loss 持续下降至 ~42，**数据量是关键因素** |
| **B2. CTMC (向量化)** | ctmc + fineweb-edu 100k | ✅ 已验证 | loss 从 12.2 → 6.0，收敛平稳，向量化 ~40-50x 加速 |
| **G1. 大数据基线** | paper loss + fineweb-edu 100k | ✅ 已验证 | loss 从 53 → 28-29，后期波动较大 |
| **G2. 大数据时间条件化** | G1 + `condition_text_on_time` | ✅ 已验证 | train_loss=31.013，与 G1 完全一致，时间条件化无效果 |

实验详细记录见 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/editflow_alignment_experiments_log.md`。

## 参考文档入口
- 里程碑定义：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_zh.md`
- 代码对齐审计：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_paper_alignment_audit_2510_03506.md`
- 归零式验证：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md`
- 过拟合排障：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_overfit_debug_zh.md`
- 训练计划：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/oneflow_text_training_plan_npu_zh.md`
- H200 执行手册：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/oneflow_h200_offline_pt_pseudo_sft_zh.md`
- EditFlow 对比分析：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/oneflow_editflow_textonly_comparison_zh.md`
- EditFlow 对齐实验记录：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/editflow_alignment_experiments_log.md`
- H200 实时进展日志：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/oneflow_h200_progress_log.md`
