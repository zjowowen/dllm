## EditFlow pipeline 与 OneFlow text-only 对比分析（可借鉴点）

本文档聚焦 `dllm/pipelines/editflow` 与 OneFlow text-only 训练/采样在功能上的相似与差异，并总结可借鉴的模型与损失设计要点。

### 1) EditFlow pipeline 关键设计

#### 1.1 对齐与噪声构造（编辑视角）
- 使用 Needleman–Wunsch 对齐 `x0/x1`，产生包含 `BLANK=-1` 的 `z0/z1`，并保证 BOS 保留（`align_with_blanks`, `BLANK`, `strip_blanks`）。  
  相关：`dllm/pipelines/editflow/trainer.py` 的 `align_with_blanks`, `BLANK`, `strip_blanks`。
- 通过 κ-mixing 逐列采样 `z_t`（从 `z0/z1` 按 `κ(t)` 混合），再 `strip_blanks` 得到 `x_t`；同时基于 `z_t` 与 `z1` 构造剩余编辑集合（`build_remaining_edits`）。  
  相关：`dllm/pipelines/editflow/trainer.py` 的 `build_remaining_edits` 与 κ-mixing 逻辑。

#### 1.2 时间与调度
- 训练时 `t ~ Unif[0, 1-ε]`，并计算 `κ(t)` 与 `w(t)=κ'(t)/(1-κ(t))`，可选 `max_w` clamp。  
  相关：`dllm/pipelines/editflow/trainer.py` 中 `t`, `k`, `w`。

#### 1.3 模型头设计（SUB/DEL/INS 三分支）
- 模型输出：`sub_logits / ins_logits`（词表分布）与 `sub/del/ins_rate_hat`（正值率）。  
  相关：`dllm/pipelines/editflow/models/qwen2/modeling_qwen2.py` 中 `sub_logits`, `ins_logits`, `rate_heads`。
- 率头使用 `Softplus` 保证正值。  
  相关：同上 `rate_heads = nn.Sequential(..., nn.Softplus())`。

#### 1.4 CTMC 风格损失（survival + positive）
- survival 项：`Lambda_hat = sum(rate_hat)`（按有效位置），`loss_surv = mean(w * Lambda_hat / denom)`。  
- positive 项：针对剩余编辑，累加 `-log rate` 与 `-log token prob`（SUB/INS）或仅 `-log rate`（DEL），最后 `loss_pos = mean(w * loss_pos_per / denom)`。  
  相关：`dllm/pipelines/editflow/trainer.py` 的 loss 计算段落。
- 归一化：默认用 `x1` 长度 `L1` 作归一化（`normalize_per_position=True`）。  
  相关：`dllm/pipelines/editflow/trainer.py` 的 `normalize_per_position` 与 `L1`。

#### 1.5 采样（τ-leap）
- 采样中使用 `tau_leap_step`：先按 `w(t)` 缩放 `rate_hat`，再采样 SUB/DEL/INS 触发。  
- 支持 `edit_prompt`（保护 prompt 不被编辑）与 `time_independent`（无 edits 时复用上次前向输出）。  
  相关：`dllm/pipelines/editflow/sampler.py` 的 `tau_leap_step` 与 `EditFLowSamplerConfig`。

#### 1.6 x0 初始化策略
- `x0` 由 `EditFlowCollator` 构造，支持空序列或 mask 序列（`empty` / `masks`），可控 prompt 保留。  
  相关：`dllm/pipelines/editflow/utils.py` 的 `EditFlowCollator` 与 `X0Sampler`。

---

### 2) OneFlow text-only 关键设计（对照）

#### 2.1 τ_text 采样与 κ-keep
- `τ_text ~ Unif[0, tau_text_max]`，`t_text = min(1, τ_text)`。  
  相关：`dllm/pipelines/oneflow/sequence_ops.py` 的 `sample_tau_text`, `tau_to_t_text`。
- κ-keep 删除策略构造 `X_t` 与 bag-of-tokens `A_i`（BOS 强制保留）。  
  相关：`dllm/pipelines/oneflow/sequence_ops.py` 的 `build_noised_xt_and_bags`。

#### 2.2 模型输出与损失
- 文本头：`pi`（零插入概率）、`lambda_nonzero`（插入率）、`q_logits`（词表分布）。  
  相关：`dllm/pipelines/oneflow/trainer.py` 中 `pi/lam/q_logits` 提取。
- 默认 `paper` loss：Eq(7) 形式（`pi` BCE + `lambda` Poisson + bag-of-tokens CE），不使用 `w(t)`；可切换 CTMC 风格（`text_loss_type="ctmc"`）。  
  相关：`dllm/pipelines/oneflow/trainer.py` 的 `text_loss_type` 分支与 `text_loss_paper_eq7_fast` 调用。
- 归一化默认按 slot 数（`normalize_text_loss_by_length=True` 时通过内部逻辑实现）。  
  相关：同上 `normalize_text_loss_by_length` 与 Eq(7) 计算路径。

---

### 3) 功能相似处与关键差异

#### 相似处
- 都基于 `κ(t)` 进行 noising/keep 的时刻采样。  
  EditFlow: `t, κ, w`；OneFlow: `τ_text → t_text → κ`。
- 都在文本端进行非自回归式“编辑/插入”建模，而非单纯 token CE。

#### 差异点（核心）
1) **编辑操作集合**  
   - EditFlow：显式建模 `SUB/DEL/INS`（三路 rate + 两路 token logits）。  
   - OneFlow text-only：插入为核心，采用 `pi + lambda + Q` 的零膨胀建模（无显式 SUB/DEL）。

2) **噪声构造方式**  
   - EditFlow：基于对齐的 `z0/z1` 与剩余编辑集合。  
   - OneFlow：基于 κ-keep 删除 + bag-of-tokens 目标。

3) **损失权重**  
   - EditFlow：使用 `w(t)=κ'(t)/(1-κ(t))` 强制权重。  
   - OneFlow paper loss：默认不使用 `w(t)`，仅在 `ctmc` 模式才使用。

4) **归一化策略**  
   - EditFlow：默认按目标长度 `L1` 归一化。  
   - OneFlow：按 slot 数（bag-of-tokens slots）归一化。

5) **时间条件**  
   - EditFlow：`t` 传入 forward（未来可接入时间嵌入）。  
   - OneFlow：默认 `condition_text_on_time=False`，文本侧时间条件恒为 0。

---

### 4) 可借鉴/复用点

1) **显式 rate 头 + Softplus**  
   - EditFlow 的 `rate_heads` 设计清晰且可解释，OneFlow 可借鉴其 rate 头的初始化与稳定性策略。  
   相关：`dllm/pipelines/editflow/models/qwen2/modeling_qwen2.py`。

2) **“rate_hat × w(t)” 结构化分离**  
   - EditFlow 将模型输出与调度权重解耦，便于控制 `w(t)` 的数值稳定；OneFlow 可在 `ctmc` 模式中对齐这一结构并做对比实验。  
   相关：`dllm/pipelines/editflow/trainer.py` 的 `w` 与 loss。

3) **x0 初始化策略**  
   - EditFlow 的 `x0` sampler（empty/masks）可以作为 OneFlow text-only 的可控初始化基线，用于调试模型是否能更快拟合。  
   相关：`dllm/pipelines/editflow/utils.py` 的 `X0Sampler` 与 `EditFlowCollator`。

4) **权重拷贝初始化（从 LM head 复用）**  
   - `init_editflow_from_src` 会将源模型 `lm_head` 权重复用到 `sub/ins` 头，减少冷启动不稳定；OneFlow 也可考虑将 `q_logits` 初始化为已有 LM 权重。  
   相关：`dllm/pipelines/editflow/utils.py` 的 `init_editflow_from_src`。

5) **τ-leap 采样策略**  
   - EditFlow 的采样流程可作为 OneFlow 的参考，用于构建更可控的非自回归调试采样器（尤其在 text-only 验证阶段）。  
   相关：`dllm/pipelines/editflow/sampler.py` 的 `tau_leap_step`。

---

### 5) 建议的对齐实验（面向 OneFlow text-only）

1) **w(t) 影响对比**  
   - OneFlow 里切换 `text_loss_type=ctmc`，与 EditFlow 的 `w(t)` 加权进行对比，看是否能降低 `loss_text_tok` 的波动。

2) **x0 初始化对比**  
   - 尝试类似 EditFlow 的 `x0` 初始化（例如全 mask 或空序列）作为 OneFlow 的替代输入路径，用于验证插入任务难度是否降低。

3) **时间条件开关**  
   - 对齐 EditFlow 的 `t` 条件化，启用 `condition_text_on_time=True`，观察 `loss_text_tok` 是否稳定下降。

---

### 6) 总结

EditFlow 在“结构化编辑”上的设计（对齐、显式 SUB/DEL/INS、CTMC 加权）对 OneFlow text-only 的调试与设计优化有较强借鉴价值，尤其在 **loss 结构、rate 头稳定性与初始化策略** 上。  
对于 OneFlow 来说，若目标是更稳定的 text-only 收敛曲线，可以优先尝试：  
`w(t)` 加权对比、时间条件化、以及引入类似 EditFlow 的 x0 初始化策略。

