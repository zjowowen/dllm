# OneFlow 文档对齐设计

## 背景

当前 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm` 中的 OneFlow 文档已经形成较完整的设计、验证、工程与实验记录体系，但不同层级文档对“当前现状”的表述已经出现偏差。尤其是状态类文档里，部分内容把“代码入口已搭好”写成了“核心逻辑已落地”或“实验已验证”，与当前代码实现和真实实验闭环不完全一致。

已确认的典型偏差包括：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md` 将 `mixed_generation_prob` 表述为主干采样/训练逻辑中的可控开关，但核心实现当前只稳定落地了 `tau_text_max` 与 `log_split_losses` 一类参数，尚未把 `mixed_generation_prob` 接成主干训练开关。
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md` 将 `tau_text_min` 写成 image-only 阶段的既有配置，但代码里当前只有 `tau_text_max` 被接入核心采样路径。
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md` 与 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md` 对 mixed/interleaved 的描述偏向“已具备独立实现”，但从代码关系看，当前更多是薄包装复用基础 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow` 主干。
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py` 仍明确限制为 `bs=1`，且 `infill` 尚未实现，文档需要稳定反映这一 v1 边界。

这次工作目标不是修改代码能力，而是让文档准确回答“现在做到哪一步、哪些已验证、哪些仍是目标或待验证”。

## 目标

建立一套一致的 OneFlow 文档口径，使读者只要阅读总览与各 track 的进展文档，就能正确理解：

- 哪些能力已经在代码里落地；
- 哪些能力只是入口、包装层或实验脚手架；
- 哪些实验已经真实跑通；
- 哪些仍属于设计目标、待验证路线或后续规划。

## 非目标

- 不修改 OneFlow 代码实现。
- 不“顺手修复” `mixed_generation_prob`、`tau_text_min`、`bs=1` 或 `infill` 等实现缺口。
- 不改写历史实验结果，不伪造未发生的验证结论。
- 不重构整套文档目录结构。

## 文档分层策略

本次对齐按四层处理，不同层承担不同职责。

### 1. 状态总览层

代表文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md`

职责：

- 给出 OneFlow 的当前主结论；
- 提供最推荐的阅读顺序；
- 概括各 track 的成熟度与依赖关系；
- 明确“代码已实现 / 已验证 / 待验证”的区别。

对齐原则：

- `README.md` 负责导航，不承载大量争议性状态细节；
- `PROGRESS.md` 作为“当前现状”的主文档，所有总结性表述应优先在这里对齐；
- 如果某条状态无法被代码或实验记录直接支持，降级为“计划”或“待验证”。

### 2. 分支进展层

代表文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md`

职责：

- 说明每条路线的目标、现状、验证缺口与下一步；
- 清楚区分“已有入口/脚本/包装层”和“已有稳定实验结论”；
- 对外表达该路线的研究优先级。

统一格式策略：

- 每个文档尽量显式包含 `目标`、`代码现状`、`已验证`、`未验证`、`下一步` 五类信息；
- checklist 只表示验证或执行状态，不再混用为“文件是否存在”的语义；
- 对存在但未验证的入口脚本，表述为“已具备入口”；
- 对尚未接入主逻辑的配置项，表述为“文档目标/实验语义，未在核心逻辑中完全落地”。

### 3. 设计与审查层

代表文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_zh.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_paper_alignment_audit_2510_03506.md`

职责：

- 保留算法目标、架构分析、代码对齐审计、验证框架；
- 说明“理论要求”和“当前代码现状”的关系；
- 为状态文档提供证据，而不是替代状态文档。

对齐原则：

- 不主动重写理论内容；
- 只修正那些把“目标设计”误写成“当前已实现事实”的语句；
- 若文档本身已清楚标注“需验证”“建议”“审查项”，则保持不动。

### 4. 历史日志与执行手册层

代表文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS_910C.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/oneflow_h200_progress_log.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/oneflow_h200_offline_pt_pseudo_sft_zh.md`

职责：

- 保存已发生实验、执行流程和平台经验；
- 提供时间线，而不是统一全局现状。

对齐原则：

- 历史结果本体不改；
- 仅在必要时补充交叉引用，提醒读者当前全局状态应以 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md` 为准。

## 事实来源与判定规则

文档对齐时采用“代码优先、实验记录次之、设计目标最后”的判定顺序。

### A. 代码优先

以下文件作为当前实现现状的一级证据：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/trainer.py`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sequence_ops.py`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_image_only/trainer.py`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_mixed_generation/trainer.py`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_interleaved/trainer.py`

判定规则：

- 如果某参数在示例入口、docstring、脚本里出现，但未被主干逻辑消费，则文档不得把它写成“已接入核心逻辑”。
- 如果某路线只是复用基础 trainer/sampler 并覆写少量默认值，则文档应写成“薄包装/实验封装”，而不是“独立实现”。
- 如果采样器或推理接口仍有限制，状态文档必须清楚列为当前边界。

### B. 实验记录次之

以下文档作为“已验证”结论的主要证据：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS_910C.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/oneflow_h200_progress_log.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS.md`

判定规则：

- “已验证”必须有实验记录、日志或测试支撑；
- 仅有脚本、入口或 TODO，不足以写成“已验证”；
- “已有单测”不等于“已有端到端实验闭环”。

### C. 设计目标最后

如果某说法来自设计文档但尚未由代码或实验验证支撑，则在状态类文档中统一降级为：

- `目标`
- `计划`
- `待验证`
- `实验语义`

而不是：

- `已支持`
- `已具备`
- `已完成`
- `已验证`

## 关键对齐项

本次文档修订必须至少覆盖以下内容。

### 1. `mixed_generation_prob` 的口径

- 在总览与 mixed/interleaved 相关文档中，明确它目前更多是入口脚本和实验语义上的参数；
- 不再把它描述为基础主干训练逻辑中的稳定混合开关；
- 如果文档保留该参数，应补一句“当前代码路径下是否真正控制样本混合，仍需进一步落地或验证”。

### 2. `tau_text_min` 的口径

- 在 image-only 文档中，删除或降级“`tau_text_min` 已存在并决定文本全保留”的既有表述；
- 改为说明当前主干可通过 `tau_text_max` 与 `t_text=min(1, tau_text)` 进入 `tau_text>1` 区间，但尚无独立 `tau_text_min` 核心配置接入。

### 3. 变体 pipeline 的实现层级

- image-only、mixed-generation、interleaved 文档中，应明确这些分支当前主要是基于基础 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow` 的复用与默认值封装；
- 不能把它们描述成已经完全分化出的独立算法实现。

### 4. 采样器限制

- 总览与 interleaved 相关文档中，应稳定写明当前采样器 v1 仍以 `bs=1` 为边界；
- 明确 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py` 的 `infill` 尚未实现；
- 若设计文档提到完整推理能力，应注明那是目标形态，不代表全部已交付。

### 5. “已实现”和“已验证”的区别

- 所有状态文档统一使用两套语言：
  - `代码现状`：描述文件、入口、脚手架、包装层、测试构件是否存在；
  - `实验现状`：描述是否真正跑出结果、是否已 overfit、是否有 decode/reconstruction 或 loss 曲线结论。

## 预期修改范围

本次优先考虑以下文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md`

必要时再补充检查：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_zh.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_paper_alignment_audit_2510_03506.md`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS_910C.md`

## 风险与控制

### 风险 1：把“设计目标”误删成“现状缺失”

控制方式：

- 设计/审查文档只修语气，不删理论结构；
- 明确使用“目标”“建议”“待验证”等词保留研究方向。

### 风险 2：把历史日志改坏

控制方式：

- 不改历史实验条目本身；
- 只在必要时补当前状态引用。

### 风险 3：多个文档再次出现不一致

控制方式：

- 以 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md` 作为统一口径源；
- 修改完成后，对关键术语进行全局复查，如 `mixed_generation_prob`、`tau_text_min`、`bs=1`、`infill`、`已验证`。

## 验收标准

完成后应满足以下标准：

1. `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md` 不再把 `mixed_generation_prob` 写成已接入主干训练/采样逻辑的既成事实。
2. `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md` 不再把 `tau_text_min` 写成现有核心配置。
3. `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md` 与 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md` 能清楚表达“当前代码多为对主干的复用与实验包装”。
4. `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md` 与核心进展文档对 OneFlow 当前成熟度给出一致叙述。
5. 涉及采样能力的文档能准确反映 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py` 的 v1 边界。
6. 文档中的“已实现”“已验证”“待验证”三类措辞不再混用。

## 实施后预期结果

完成这轮文档对齐后，新的贡献者应能够基于文档快速形成以下正确认知：

- 基础 OneFlow 主干已经具备较完整的训练与采样框架；
- text-only 是当前实验上最成熟的一条线；
- image-only、mixed-generation、interleaved 的主要缺口不在“有没有入口”，而在“有没有真实实验闭环”；
- 下一步工作的优先级应该建立在上述现状之上，而不是建立在过度乐观的文档口径之上。
