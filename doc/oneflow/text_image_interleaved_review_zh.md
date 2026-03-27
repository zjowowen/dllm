# OneFlow Text-Image Interleaved 架构审查与实验设计

> **背景**：`oneflow_text_only` 的控制变量实验仍在进行中，已确认 CTMC loss 对训练稳定性的关键改善作用（见 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/losses_design_zh.md` 与 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS_experiment.md`）。本文档在此基础上，提前对 text-image interleaved（多模态）的架构进行系统审查，并设计整体实验方案。
>
> **关联文档**：
> - 算法设计总览：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_zh.md`
> - 论文规格摘录：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_paper_spec_2510_03506.md`
> - 代码对齐审计：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_paper_alignment_audit_2510_03506.md`
> - 归零式验证手册：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md`
> - Transfusion 三仓对比：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/engineering/transfusion_three_repo_comparison_zh.md`
> - Text-only 实验进展：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS_experiment.md`
> - Text-only Loss 设计解析：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/losses_design_zh.md`

---

## 第一部分：架构审查

---

## 1. 训练架构审查

### 1.1 Interleaved Schedule（交错时间表）

**论文定义（Algorithm 3, Section 2.3.2）**：

```
τ_text ~ Unif[0, 2]
t_text = min(1, τ_text)

对每张图像 i：
  u_i ~ Unif(0, 1)
  τ_img_i = τ_text - κ^{-1}(u_i)
  若 τ_img_i < 0：图像尚未插入，<|oneflow_image|> 作为被删 token 进入 bag
  否则：t_img_i = min(1, τ_img_i)，训练 flow matching
```

**代码实现（`sequence_ops.py: apply_interleaved_image_schedule()`）**：

| 审查点 | 论文要求 | 代码现状 | 状态 | 风险/备注 |
|--------|----------|----------|------|-----------|
| τ_text 范围 | Unif[0, 2] | `tau_text_max=2.0`，可配置 | ✅ | 默认对齐 |
| t_text 截断 | min(1, τ_text) | `t_text = min(tau_text, 1)` | ✅ | — |
| τ_img 计算 | τ_text - κ^{-1}(u) | 实现了 `kappa_inverse` | ✅ | 需验证 κ^{-1} 在非线性调度下的数值精度 |
| τ_img < 0 删除 | <\|image\|> 进入 bag | 删除 token 并合并 bag | ✅ | **关键点**：删除后需要右到左 bag 合并，已实现 |
| τ_img ≥ 0 保留 | t_img = min(1, τ_img) | `t_img = min(1.0, tau_img)` | ✅ | — |
| BOS 强制保留 | 隐含 | 代码强制保留 BOS | ✅ | — |
| prompt_len 保留 | 论文未指定 | 若提供则保留 prompt 前缀 | ✅ | SFT 场景需要 |

**重点审查项 A：τ_text > 1 时的行为**

当 τ_text ∈ (1, 2] 时，t_text = 1，意味着**按当前调度语义应当是所有文本 token 都被保留**（κ(1) = 1），文本侧不做删除。论文将该区间作为 "mixed generation" / image-only 过渡语义；但对当前仓库而言，这仍应视为**待验证的目标行为**，不能直接当作已充分验收的既成事实。

- **需验证**：当 t_text = 1 时，文本 loss 的分母 n = len(X_t) = len(X_1)，且所有 bag 为空（k_i = 0 ∀i）。此时：
  - Paper Eq7：loss_tok = 0, loss_lam = 0, loss_pi = Σ BCE(π_i, 1) → 推动 π→1（正确）
  - CTMC：w(1) = κ'(1)/(1-κ(1)) → 发散（⚠️ 被 max_w 截断，但需确认截断值是否合理）
- **实现口径说明**：`tau_text_max` 可以表达论文中的上界语义，但 `tau_text_min` 目前还不是已完全打通的核心运行时配置；同理，这里的 mixed/image-only 讨论应优先理解为实验目标语义，而不是“基础路径已经稳定支持所有阶段控制”。

**重点审查项 B：图像删除后的 bag 合并语义**

当一张图像的 `<|oneflow_image|>` token 被删除时：
1. 该 token 本身需加入前一个 slot 的 bag
2. 该 token 之后原有的 bag 内容需合并到前一个 bag

这是 interleaved schedule 最容易出错的环节。当前代码采用**右到左遍历**处理多图删除，避免索引错乱。

- **需验证**：多图场景下连续删除是否正确（例：3 张图像中删除第 1、3 张）

**重点审查项 C：κ^{-1} 在非线性调度下的精度**

论文推荐线性调度 κ(t) = t，此时 κ^{-1}(u) = u，无精度问题。若使用 cosine 或高次调度：
- κ^{-1} 通过解析式或数值二分实现
- 需关注 u ∈ {0, 1} 边界的数值稳定性

### 1.2 统一序列构建（Unified Sequence）

**训练时（`sequence_ops.py: build_unified_train_batch()`）**：

| 审查点 | 要求 | 代码现状 | 状态 |
|--------|------|----------|------|
| 文本 token 与图像 latent token 交错排列 | 图像 latent 插在对应 <\|oneflow_image\|> 之后 | `xt_to_total_pos_list` 记录映射 | ✅ |
| `is_any_modality` 掩码 | 图像位置为 True，文本为 False | 逐位置标注 | ✅ |
| `modality_positions` 元数据 | (type, offset, length) | 记录每张图像的偏移和长度 | ✅ |
| `times` 向量 | 文本用 t_text（或常数），图像用 t_img | 分模态赋值 | ✅ |
| flow_targets | 图像位置：Y1 - Y0；文本位置：零 | 训练时构建 | ✅ |
| Padding 对齐 | batch 内不同长度对齐 | pad_id 填充 + attention_mask | ✅ |

**风险项**：

1. **序列长度膨胀**：每张图像占 `image_num_tokens`（如 256）个位置。多图样本的统一序列可能很长 → 显存压力。
   - 需评估：在 910C/H200 上，text(~128) + 1 image(256) + padding 是否在显存预算内
   - 多图（2-3 张）场景的可行 batch size

2. **`xt_positions` 在 loss 计算中的映射**：混模态时，文本 loss 需要通过 `xt_positions` 将文本 slot 映射回统一序列中的实际位置。映射错误将导致 loss 计算取到图像位置的 head 输出。

3. **rotary position 计算**：`derive_rotary_positions_from_modality_positions()` 需确保文本 token 的相对位置不被图像 block 打断。当前 Transfusion 的做法是：图像 block 内部用 axial position，文本 token 的全局位置按"跳过图像 block"计算。

### 1.3 损失函数

**当前实现的两条文本损失路径**：

| 路径 | 数学形式 | 训练稳定性 | 论文对齐 |
|------|----------|------------|----------|
| Paper Eq7 | token CE + π BCE + λ 零截断 Poisson，**不乘** w(t) | ⚠️ 不稳定（text_only 实验已确认：90x 方差） | ✅ 论文原始形式 |
| CTMC | survival + positive，**乘** w(t) = κ'/(1-κ) | ✅ 稳定（1.2x 方差） | ≈ 论文前作 Edit Flows 形式 |

**关键发现（来自 text_only 实验，`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS_experiment.md`）**：

Paper Eq7 的不稳定根因：
1. 归一化分母 n（n_slots）在 t→1 时趋近 1，导致 CE 累加爆炸
2. 缺失 w(t) 权重放大了不同时间步的方差
3. λ→0 时零截断 Poisson NLL 数值不稳定

**对多模态的影响**：

- 当 τ_text > 1 时，文本完全保留，text loss 主要来自 π BCE → 此区间 Eq7 的 n_slots 不会缩小，不会触发根因 1
- 当 τ_text ∈ [0,1] 时，text + image 同时训练，text loss 不稳定会干扰整体优化
- **建议**：多模态训练也应优先使用 CTMC loss（至少作为基线对照之一）

**图像 loss（Flow Matching, Eq. 9）**：

```
L_img = (1/N_img) Σ_{j∈img-pos} ||v_j - (Y1_j - Y0_j)||²
```

| 审查点 | 要求 | 代码现状 | 状态 |
|--------|------|----------|------|
| MSE 监督 | v 对齐 flow = Y1 - Y0 | `image_loss_flow_matching()` | ✅ |
| 掩码过滤 | 仅图像位置计算 | `is_any_modality` 掩码 | ✅ |
| 归一化 | 按图像 token 数 | `normalize_image_loss_by_tokens=True` | ✅ |
| 空图像 batch | 返回零 loss | 有分支处理 | ✅ |

**总损失组合**：

```
L_total = L_text + image_loss_weight × L_img
```

- `image_loss_weight` 默认 1.0
- ⚠️ **需实验确定**：text loss（CTMC ~12）与 image loss（MSE 量级依赖 latent scale）的数值尺度差异。若差距过大，需调整权重或归一化策略。

### 1.4 模型架构

**当前架构（`OneFlowModel`）**：

```
输入 → text_embed / latent_to_model → 混合嵌入（where mask）
    → TransfusionTransformer (unified trunk, modality-aware attention)
    → 四个 head：
        to_pi      → sigmoid → π [B, N]
        to_lambda   → softplus → λ [B, N]
        to_q_logits → logits  → Q [B, N, V]
        to_v       → linear  → v [B, N, dim_latent]
```

| 审查点 | 状态 | 备注 |
|--------|------|------|
| 文本/图像嵌入混合 | ✅ | `torch.where(is_any_modality, mod_emb, text_emb)` |
| 模态感知注意力掩码 | ✅ | 支持 manual attn_mask（vendored 改动） |
| Per-token 时间条件 | ✅ | 文本/图像各自 time，通过 trunk 的 AdaptiveWrapper 调制 |
| 图像 head 输出维度 | ✅ | `dim_latent`（默认 4，对应 SD VAE） |
| 文本 head 与图像位置的隔离 | ⚠️ | head 输出全序列，需依赖外部 mask 选取正确位置 |

**风险项**：

1. **Head 输出全序列的效率**：四个 head 对全序列计算，但图像位置只用 v，文本位置只用 π/λ/Q。浪费了计算量。
   - 短期可接受，长期可考虑 conditional head（按 mask 分别计算）

2. **latent_to_model 维度不匹配**：当 `dim_latent ≠ dim` 时，使用 `nn.Linear` 投影。需确保梯度正常流过此投影层。

3. **attention mask 的语义**：
   - 文本：默认双向（非 AR，符合 Edit Flow 语义）
   - 图像 block 内部：双向
   - 文本→图像、图像→文本：当前是否允许互相 attend？论文未明确规定。
   - **建议**：初始实验用全双向，后续可尝试图像 block 对文本的单向注意力

### 1.5 数据流完整链路

```
WDS sample (text + image latent)
  ↓
OneFlowCollator → {x1_ids, image_latents, prompt_len}
  ↓
Sample τ_text ~ Unif[0,2], t_text = min(1, τ_text)
  ↓
κ-keep noising → X_t + bags_list
  ↓
Interleaved image schedule (τ_img = τ_text - κ^{-1}(u))
  ├─ τ_img < 0 → 删除 <|image|>, 合并 bag
  └─ τ_img ≥ 0 → 保留，构造 Y_t = t_img*Y1 + (1-t_img)*Y0
  ↓
build_unified_train_batch → {input_ids, is_any_modality, modality_tokens, modality_positions, times, flow_targets}
  ↓
OneFlowModel.forward → {π, λ, Q, v}
  ↓
Loss: text_loss(π,λ,Q | bags, xt_positions) + w_img × image_loss(v | flow_targets, is_any_modality)
  ↓
Backward + Optimize
```

---

## 2. 推理架构审查

### 2.1 采样循环（Algorithm 1-2）

**当前实现（`sampler.py: OneFlowSampler.sample()`）**：

```
初始化：X = [BOS] (或 prompt), images = [], t_text = 0

每步（dt 固定，如 0.05）：
  1. 构建统一序列：interleave X 与 images 的 latent tokens
  2. Model forward → π, λ, Q, v
  3. 图像 Euler 更新：Y ← Y + dt_img × v, t_img ← t_img + dt_img
  4. 文本插入：
     - p_lam = dt × w(t_text) × λ
     - p_pi = 1 - π (可选 gate)
     - do_insert = Bernoulli(p_lam) AND Bernoulli(p_pi)
     - 若插入 <|oneflow_image|>：创建新图像 Y~N(0,I), t_img=0
  5. t_text ← t_text + dt

终止：t_text ≥ 1 且所有 t_img ≥ 1
```

| 审查点 | 论文要求 | 代码现状 | 状态 |
|--------|----------|----------|------|
| 图像 Euler 更新 | ΔY = dt × v | 按 image slice 更新 | ✅ |
| 插入概率 p^λ | dt × w(t) × λ | `p_lam = dt * w(t) * lam`，有 clamp | ✅ |
| π gate | 可选 | `use_pi_gate=True` 默认开启 | ✅ |
| 新图像创建 | 采样到 <\|image\|> 时创建 | `images.insert(...)` | ✅ |
| 终止条件 | 全部 t ≥ 1 | `t_text >= 1-ε and all(t_img >= 1-ε)` | ✅ |
| BS=1 限制 | — | `len(inputs) != 1` 时 raise | ⚠️ 已知限制 |
| KV-cache | — | 支持但未启用 | ⚠️ 性能优化空间 |

**风险项**：

1. **dt 固定步长的粗糙性**：论文使用固定 dt，但未讨论自适应步长。当图像接近 t_img=1 时，小步长更精确但计算量增大。
   - 短期用固定 dt（0.05 → 20 步），足够实验
   - 长期可考虑自适应 ODE solver（如 Heun、RK4）

2. **插入概率 clamp**：当 dt 较大时 p_lam 可能 > 1，需 clamp 到 [0, 1]。当前已实现。

3. **多图生成的动态增长**：每次插入 `<|oneflow_image|>` 都会新增一张图像 → 序列长度持续增长 → 可能超出 max_seq_len。
   - 已有 `max_seq_len` 和 `max_new_tokens` 限制
   - 需实验中设置合理上限

4. **文本插入顺序（右到左）**：当前实现按右到左应用插入，避免索引偏移。这与论文的"并行插入"语义一致（一步内所有插入基于相同状态计算概率，然后一次性应用）。

### 2.2 图像解码

采样结束后，每张图像的 latent Y（shape `[N, dim_latent]`）需 reshape 并送入 VAE decoder：

```
Y [N, 4] → reshape [4, H, W] → VAE decode → PIL Image
```

| 审查点 | 要求 | 代码现状 | 状态 |
|--------|------|----------|------|
| latent reshape 约定 | row-major: [N,4] → [4,H,W] | 在外部脚本处理 | ✅ |
| latent_scale 一致 | 训练/推理用同一 scale（0.18215） | 配置化 | ✅ |
| VAE decoder | 标准 SD VAE | 外部调用 | ✅ |

---

## 3. 训练-推理一致性审查

这是最容易出现隐蔽 bug 的区域。训练和推理对同一套 head 输出的使用方式必须数学上一致。

### 3.1 核心一致性矩阵

| 维度 | 训练侧 | 推理侧 | 一致性 | 风险说明 |
|------|--------|--------|--------|----------|
| **π 的语义** | BCE(π, 1[k=0])：π 预测"该位置无插入"的概率 | p_pi = 1 - π：插入概率 = 1 - π | ✅ 一致 | — |
| **λ 的语义** | Poisson rate（非零计数强度） | p_lam = dt × w(t) × λ_nonzero | ✅ 一致 | 训练不乘 w(t)（Eq7），但推理乘 w(t)；**这是论文的设计意图**——w(t) 从 loss 中 factor out，在采样时回到 rate 中 |
| **Q 的语义** | bag-of-tokens CE | a ~ Q(·) 采样 | ✅ 一致 | 训练对 bag 中所有 token 计算 CE；推理时每次只采 1 个 token |
| **v 的语义** | MSE(v, Y1-Y0) | Y ← Y + dt × v | ✅ 一致 | 训练监督 velocity field，推理做 Euler 积分 |
| **time conditioning** | `condition_text_on_time=False`（默认）：text times 常数 | 同配置 | ✅ | **必须训练/推理用同一设置** |
| **图像 time** | 图像 token 的 times = t_img | 图像 token 的 times = t_img | ✅ | — |
| **序列构建** | `build_unified_train_batch()` | `build_unified_sampler_inputs_bs1()` | ⚠️ | 两个函数逻辑独立，需确保位置映射、modality_positions、is_any_modality 的语义完全一致 |
| **attention mask** | 训练用 padding mask + modality-aware | 推理用同规则 | ⚠️ | 推理时序列动态增长，mask 需随之更新 |

### 3.2 关键不对称性（by design）

以下不对称是论文刻意设计的，不是 bug：

1. **训练不乘 w(t)，推理乘 w(t)**（仅 Paper Eq7 路径）：
   - 论文将 w(t) 从 loss 中 factor out（称"不影响最优解"）
   - 推理时 rate = w(t) × λ，这是 CTMC 采样的正确形式
   - **但若使用 CTMC loss（训练时乘了 w(t)）**：需确认推理时是否应再乘一次 w(t)。答案是**是**——CTMC loss 的 w(t) 是训练权重，采样率仍需要 w(t) × λ

2. **训练时 bag 有多个 token，推理时每次插入 1 个**：
   - 训练：每个 slot 的 bag 可能包含多个被删 token → CE 对 bag 中每个 token 独立计算
   - 推理：每步每个 slot 最多插入 1 个 token（Bernoulli 触发后采样 1 个 a ~ Q）
   - 这符合论文的 CTMC 采样语义：连续时间极限下，每步最多插入一个

3. **训练时 noising 随机，推理时步进确定**：
   - 训练：每个 batch 随机采样 τ_text、u_i，一次性构建快照
   - 推理：从 t=0 确定性步进到 t=1

### 3.3 需要重点验证的一致性项

**项 1：统一序列构建的训练/推理一致性**

训练用 `build_unified_train_batch()`，推理用 `build_unified_sampler_inputs_bs1()`。两者独立实现，需确保：
- 图像 latent token 在统一序列中的位置（相对于 `<|oneflow_image|>` 锚点 token）一致
- `modality_positions` 的 (type, offset, length) 格式一致
- `is_any_modality` 的赋值规则一致
- `times` 的赋值规则一致

**验证方法**：构造一个"训练时 t=0（全部保留）"的样本，比较训练侧和推理侧生成的统一序列是否完全一致。

**项 2：推理时新增图像的处理**

当推理过程中采样到 `<|oneflow_image|>` token 时：
- 新建一张图像 latent Y ~ N(0,I)
- 设 t_img = 0
- 下一步需要在统一序列中为新图像预留 `image_num_tokens` 个位置

⚠️ **风险**：训练时 `image_num_tokens` 由 VAE encoder 决定（如 16×16=256），推理时由配置指定。两者必须一致。

**项 3：`condition_text_on_time` 配置**

这是一个训练/推理必须完全对齐的开关：
- 训练 `False` + 推理 `True` → 推理时 text 看到了训练中没有的时间信息 → 行为不可预测
- 训练 `True` + 推理 `False` → 推理时 text 丢失了时间信息 → 退化但不崩溃

当前默认均为 `False`，一致。但根据 text_only 实验，CTMC + time conditioning 可能更优 → 若多模态训练采用 time conditioning，推理也必须开启。

---

## 第二部分：实验设计

---

## 4. 实验总体策略

### 4.1 指导原则

1. **渐进式验证**：从单模态 → 混合模态 → 全交错生成，每一步都有明确的验收标准
2. **继承 text_only 发现**：优先使用 CTMC loss 作为文本侧基线，与 Paper Eq7 做对照
3. **控制变量**：每次实验只改一个关键变量
4. **可复现**：固定 seed、记录完整配置、输出 trace JSON

### 4.2 实验分阶段

```
Phase 0: 归零验证（数据 + 单元测试）
  ↓
Phase 1: 单模态基线建立
  ├─ 1a: Text-only（继承 oneflow_text_only 结论）
  └─ 1b: Image-only flow matching
  ↓
Phase 2: 混合模态训练
  ├─ 2a: 固定比例 mixed generation
  └─ 2b: 全 interleaved（τ_text ~ Unif[0,2]）
  ↓
Phase 3: 训练-推理一致性闭环
  ├─ 3a: Text-only 生成质量
  ├─ 3b: Image-only 生成质量
  └─ 3c: Interleaved 生成（文本 + 图像）
  ↓
Phase 4: 消融与优化
  ↓
Phase 5: 规模化
```

---

## 5. Phase 0：归零验证

**目标**：确保数据管线和核心模块无 bug，对应 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md` 的 Stage 0-5。

| 子项 | 内容 | 验收标准 | 对应命令 |
|------|------|----------|----------|
| 0-1 | WDS latent decode | GT 图像可识别，非噪声/全黑 | `verify_wds_latents_decode.py` |
| 0-2 | Text loss 数值一致 | ref vs fast 版 loss 误差 < 1e-5 | `test_oneflow_text_loss_eq7.py` |
| 0-3 | Image flow matching | oracle v=flow 时 loss≈0 | `test_oneflow_image_flow_matching.py` |
| 0-4 | Unified 序列拼接 | xt_to_total_pos 严格递增，is_any_modality 与 modality_positions 对齐 | `test_oneflow_sequence_ops.py` |
| 0-5 | Interleaved schedule | 强制触发 τ_img<0 / ≥0 两种 case，验证 bag 合并 | `test_oneflow_interleaved_schedule.py` |
| 0-6 | CPU 集成 | forward + backward 不崩，loss 有限 | `test_oneflow_integration_cpu.py` |

---

## 6. Phase 1：单模态基线

### 6.1 Phase 1a：Text-only 基线

**目的**：将 `oneflow_text_only` 的结论迁移到 `oneflow` pipeline（Transfusion trunk）上验证。

**实验矩阵**：

| 编号 | text_loss_type | condition_text_on_time | 数据 | 预期 |
|------|---------------|----------------------|------|------|
| 1a-1 | paper | False | fineweb-edu 100k | 基线（预期 loss ~28-30，波动大） |
| 1a-2 | ctmc | True | fineweb-edu 100k | 预期 loss ~12→稳定下降 |
| 1a-3 | ctmc | False | fineweb-edu 100k | 消融：time cond 的影响 |

**配置**：
```bash
# 1a-2 示例（推荐基线）
accelerate launch ... examples/oneflow/pt_text.py \
  --text_loss_type ctmc \
  --condition_text_on_time True \
  --normalize_text_loss_by_length True \
  --max_steps 1000 \
  --per_device_train_batch_size 8
```

**验收标准**：
- CTMC loss 在 Transfusion trunk 上也保持稳定（方差 < 2x）
- Loss 收敛趋势与 `oneflow_text_only`（DDiT trunk）可比

### 6.2 Phase 1b：Image-only Flow Matching 基线

**目的**：孤立验证图像侧 flow matching 训练是否正确。

**方法**：保留该实验矩阵，但将其视为 **image-only 目标语义的实验入口**：可优先使用专门的 image-only 训练入口，或通过受控实验把 τ_text 推到文本全保留区间。不要把 `mixed_generation_prob=1.0` 直接解读为“基础 oneflow 主入口已被证明能稳定控制 image-only 核心路径”；`tau_text_min` 相关下界语义也仍待补齐为核心配置。

**实验矩阵**：

| 编号 | 数据 | image_loss_weight | 步数 | 验收 |
|------|------|------------------|------|------|
| 1b-1 | flower32 (1 shard) | 1.0 | 5000 | overfit：image loss 持续下降 |
| 1b-2 | flower32 (1 shard) | 10.0 | 5000 | 对比权重影响 |
| 1b-3 | flower32 (全量) | 1.0 | 1000 | 多样本下 loss 趋势 |

**验收标准**：
- Image loss 在 5000 步 overfit 后显著下降
- 使用训练样本做 reconstruction：采样结果与 GT 有语义关联（VAE decode 后）
- 训练后采样生成的 latent 解码为非噪声图像

---

## 7. Phase 2：混合模态训练

### 7.1 Phase 2a：固定比例混合训练

**目的**：验证 text loss + image loss 能同时优化且不互相干扰。

**方法**：保留 `mixed_generation_prob` 作为入门级 mixed-generation 实验语义，用来表达“希望多少样本走含图像分支”的目标；但当前不应把它表述成**已经充分验证的基础主路径控制开关**，其实际生效点与训练闭环仍需单独实现/验收。

**实验矩阵**：

| 编号 | text_loss_type | mixed_gen_prob | image_loss_weight | 数据 | 步数 |
|------|---------------|----------------|------------------|------|------|
| 2a-1 | ctmc | 0.5 | 1.0 | flower32 + fineweb | 2000 |
| 2a-2 | ctmc | 0.5 | 5.0 | flower32 + fineweb | 2000 |
| 2a-3 | ctmc | 0.2 | 1.0 | flower32 + fineweb | 2000 |
| 2a-4 | paper | 0.5 | 1.0 | flower32 + fineweb | 2000 |

**关键观测指标**：
- `loss_text` 与 `loss_img` 分别的趋势（需开启 `log_split_losses=True`）
- 两种 loss 是否会互相"拉扯"（一种下降另一种上升）
- 梯度范数的稳定性

**验收标准**：
- text loss 和 image loss 均呈下降趋势
- 总 loss 无爆炸/NaN
- 梯度范数无持续发散

### 7.2 Phase 2b：全 Interleaved 训练

**目的**：验证完整的 τ_text ~ Unif[0,2] interleaved schedule。

**方法**：目标是转向 Algorithm 3 的完整 interleaved 逻辑；这里的表述强调的是**目标阶段语义**，不是说 `mixed_generation_prob` 或其它阶段性控制已经在基础实现里被完全替代并验证完毕。

**实验矩阵**：

| 编号 | text_loss_type | τ_text_max | image_loss_weight | 数据 | 步数 |
|------|---------------|-----------|------------------|------|------|
| 2b-1 | ctmc | 2.0 | 1.0 | flower32 + fineweb | 5000 |
| 2b-2 | ctmc | 2.0 | 动态（按 text/img token 比例） | flower32 + fineweb | 5000 |
| 2b-3 | paper | 2.0 | 1.0 | flower32 + fineweb | 5000 |

**关键观测指标**：
- τ_text > 1 时段的图像 loss 曲线
- τ_text ∈ [0,1] 时段的文本 + 图像 loss 曲线
- 图像删除/保留的比例统计（通过 trace 收集）

**验收标准**：
- Loss 无异常（特别关注 τ_text ∈ (0.9, 1.1) 过渡区间）
- 图像删除率统计上符合预期：P(τ_img < 0) 随 τ_text 下降而增大

---

## 8. Phase 3：训练-推理一致性闭环

### 8.1 Phase 3a：Text-only 采样验证

**目的**：确认训练后的模型能通过 sampler 生成有意义的文本。

**方法**：使用 Phase 1a 或 2a 的 checkpoint 运行纯文本采样。纯文本可用轻量入口 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/sample.py`；涉及图像或三模式切换时，优先使用 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py`。

```bash
python examples/oneflow/sample.py \
  --model_dir <checkpoint> \
  --prompt "OneFlow is a generative model" \
  --max_steps 20 \
  --dt 0.05 \
  --use_pi_gate True
```

**验收标准**：
- 生成的文本有基本的连贯性（对 fineweb 数据训练的模型）
- 序列长度随步数增长（插入在发生）
- π gate 的行为合理：后期步数插入概率应下降

### 8.2 Phase 3b：Image-only 采样验证

**目的**：确认图像 latent 生成链路完整。

**方法**：给定固定 prompt（如 `"a photo of a flower <|oneflow_image|>"`），运行采样并 VAE decode。

```bash
python -u examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir <checkpoint> \
  --mode image_conditioned \
  --prompt "a photo of a flower" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --max_steps 20 \
  --dt 0.05 \
  --output_dir outputs/interleaved_image_check
```

**验收标准**：
- 图像 latent 从纯噪声逐步收敛（latent 范数随步数变化合理）
- VAE decode 后的图像非纯噪声
- Overfit 模型的重建结果与训练样本有语义关联

### 8.3 Phase 3c：Interleaved 采样验证

**目的**：验证文本和图像交错生成的完整链路。

**方法**：从 BOS 开始，允许模型自由生成文本和图像。

```bash
python -u examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir <checkpoint> \
  --mode interleaved \
  --prompt "a photo of a flower" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --max_steps 40 \
  --dt 0.05 \
  --max_new_tokens 256 \
  --max_seq_len 512 \
  --output_dir outputs/interleaved_full_check
```

**关键观测**：
- 模型是否会在合适位置插入 `<|oneflow_image|>` token
- 图像 latent 是否在后续步骤中被正确 denoise
- 终止条件是否正常触发（t_text ≥ 1 且所有 t_img ≥ 1）
- 序列长度增长是否合理（不爆炸式增长）

**验收标准**：
- 完整生成链路不崩溃
- 能产出至少 1 张可 decode 的图像 + 若干文本 token
- 终止条件正常触发

---

## 9. Phase 4：消融实验

基于 Phase 2-3 的最佳配置，进行关键消融：

### 9.1 文本 Loss 选择消融

| 变量 | 选项 | 预期影响 |
|------|------|----------|
| text_loss_type | paper vs ctmc | 训练稳定性（已在 text_only 确认，需在多模态下复验） |
| normalize_text_loss_by_length | True vs False | Loss scale 对多模态平衡的影响 |

### 9.2 图像 Loss 权重消融

| 变量 | 选项 | 预期影响 |
|------|------|----------|
| image_loss_weight | 0.1, 1.0, 5.0, 10.0 | 文本/图像质量 trade-off |
| normalize_image_loss_by_tokens | True vs False | 多图时 loss 归一化策略 |

### 9.3 时间条件化消融

| 变量 | 选项 | 预期影响 |
|------|------|----------|
| condition_text_on_time | True vs False | 长期训练中文本质量（text_only 短期实验无影响，但长期可能重要） |

### 9.4 调度消融

| 变量 | 选项 | 预期影响 |
|------|------|----------|
| scheduler | linear vs cosine | 论文推荐 linear，但需在多模态下验证 |
| τ_text_max | 1.0 vs 2.0 | 1.0 = 无纯图像阶段；2.0 = 论文默认 |
| dt（推理） | 0.02, 0.05, 0.1 | 采样质量 vs 速度 |

### 9.5 Attention Mask 消融

| 变量 | 选项 | 预期影响 |
|------|------|----------|
| 文本-图像互 attend | 全双向 vs 图像→文本单向 | 论文未明确，需实验 |

---

## 10. Phase 5：规模化路线

### 10.1 硬件评估

| 配置 | text seq | image tokens/张 | 张数 | 总序列长 | 预估 BS（16 NPU） |
|------|---------|----------------|------|---------|-------------------|
| text-only | 1024 | 0 | 0 | 1024 | ≤32（910C 已验证） |
| text+1img | 128 | 256 | 1 | ~384 | 待测 |
| text+2img | 128 | 256 | 2 | ~640 | 待测 |
| text+4img | 128 | 256 | 4 | ~1152 | 待测 |

### 10.2 规模化步骤

1. **单机 16 NPU smoke**（100 步）→ 确认 OOM 边界和稳定 BS
2. **单机 16 NPU 基线**（1000-5000 步）→ 建立 loss 曲线基线
3. **2 节点 bring-up**（200-500 步）→ 验证通信和梯度同步
4. **多机长跑**（10k+ 步）→ 收敛趋势和生成质量
5. **SFT**（Phase 5b）→ 指令微调和对话式多模态生成

---

## 11. 评估方案

### 11.1 训练侧指标

| 指标 | 说明 | 来源 |
|------|------|------|
| loss_total | 总训练 loss | trainer log |
| loss_text / loss_img | 分模态 loss | `log_split_losses=True` |
| loss_tok / loss_pi / loss_lam | 文本 loss 分量 | trainer log（扩展） |
| grad_norm | 梯度范数 | trainer log |
| image_delete_ratio | 图像被删除的比例 | trace 统计 |

### 11.2 生成质量指标

| 指标 | 说明 | 适用阶段 |
|------|------|----------|
| Text prompt pass rate | 分级 prompt 通过率 | Phase 3a |
| Image FID | 生成图像与 GT 的分布距离 | Phase 3b（需足够样本量） |
| Image-text alignment | CLIP score | Phase 3c |
| Reconstruction MSE | 训练样本 latent 重建误差 | Phase 2（overfit 场景） |

### 11.3 一致性检查指标

| 指标 | 说明 |
|------|------|
| 训练/推理统一序列 diff | 固定输入下两个 builder 的输出差异 |
| π 分布统计 | 训练时 π 的预测分布 vs 推理时的行为 |
| 插入率 vs 训练删除率 | 推理每步插入 token 数 vs 训练对应 t 的删除数 |

---

## 12. 已知风险与缓解策略

| 风险 | 严重度 | 缓解策略 |
|------|--------|----------|
| Paper Eq7 文本 loss 不稳定 | 高 | 默认使用 CTMC loss；若需 Eq7 结果，配合 grad clip |
| Text/image loss 量级不匹配 | 中 | Phase 2a 通过 image_loss_weight 消融确定最优值 |
| 多图场景序列长度爆炸 | 中 | 限制 max_seq_len + 减小 image_num_tokens（更小 latent 分辨率） |
| 训练/推理统一序列构建不一致 | 高 | Phase 0 的单测 + Phase 3 的端到端闭环 |
| 910C OOM（多模态） | 中 | Phase 5.1 先测 OOM 边界 |
| κ^{-1} 数值精度（非线性调度） | 低 | 先用 linear（κ=t），后续再尝试 cosine |
| 新图像创建导致序列动态增长（推理） | 中 | max_seq_len / max_new_tokens 硬限制 |
| w(t) 在 t→1 时发散（CTMC） | 中 | max_w 截断（默认 20.0），需验证截断值是否合理 |

---

## 13. 优先级与建议执行顺序

```
[立即] Phase 0：归零验证
  - 跑通所有 Stage 0-5 单测
  - 特别关注 Stage 3（unified 序列拼接）和 Stage 4（interleaved schedule）

[短期] Phase 1a：Text-only 基线
  - 将 oneflow_text_only 的 CTMC 结论在 Transfusion trunk 上复验
  - 预计 1-2 天

[短期] Phase 1b：Image-only 基线
  - 单 shard overfit 验证 image flow matching 正确性
  - 预计 1-2 天，可与 1a 并行

[中期] Phase 2a：固定比例混合训练
  - 关键节点：确认 text + image loss 不互相干扰
  - image_loss_weight 初步调优

[中期] Phase 3a-3c：训练-推理闭环
  - 端到端生成验证，从 overfit 模型开始

[按需] Phase 2b：全 Interleaved 训练
  - 在 2a 稳定后开展

[按需] Phase 4：消融实验
  - 基于前面阶段的最优配置展开

[长期] Phase 5：规模化
  - 等消融完成、最优配置确定后
```

---

## 14. 附录：与 text_only 实验的关键结论对接

| text_only 发现 | 对多模态的影响 | 行动项 |
|----------------|---------------|--------|
| CTMC loss 方差仅 1.2x（vs Eq7 的 90x） | 多模态训练中文本侧 loss 波动将更小 → 与 image loss 的梯度竞争更平稳 | Phase 1a 复验，Phase 2a 默认使用 CTMC |
| Paper Eq7 归一化分母 n 在 t→1 时导致 loss 爆炸 | 多模态中 τ_text∈(1,2] 时 t_text=1，n=原始序列长度，此区间不会触发该问题；但 τ_text∈[0,1) 时仍存在 | Phase 2b 需关注过渡区间 |
| Time conditioning 短期无影响 | DDiT 的 adaLN 初始化为零导致；Transfusion trunk 的 AdaptiveWrapper 可能不同 → 需在 Transfusion 上重新验证 | Phase 1a-3 消融 |
| 数据量是关键因素（12.8k 不足，100k 可收敛） | 多模态数据（image-text pairs）通常量更大 → 数据侧约束可能更宽松 | 准备足够的 image-text 训练数据 |
| RoPE 对齐对 loss 无直接影响 | Transfusion trunk 的 rotary 已独立实现，不受影响 | 无 |
