# OneFlow 代码对齐审计（arXiv:2510.03506）

论文：[arXiv:2510.03506](https://arxiv.org/html/2510.03506)  
论文规格摘录见：`doc/oneflow/design/oneflow_paper_spec_2510_03506.md`

基线训练入口：`examples/oneflow/pt_wds_latents.py`

---

## 1) Trainer（Algorithm 3）对齐矩阵

代码位置：`dllm/pipelines/oneflow/trainer.py`（`OneFlowTrainer.compute_loss`）

| 论文要求 | 代码现状 | 状态 | 备注 |
|---|---|---|---|
| 采样 \(\\tau_{text}\\in[0,2]\)，并令 \(t_{text}=\\min\\{1,\\tau_{text}\\}\) | 支持 `mixed_generation_prob=p`：`τ_text∼Unif[1,2]`（prob=p）否则 `Unif[0,1]`；`t_text=min(τ_text,1)` | ✅ 对齐（按论文 Sec 3.0.1） | 论文 Appendix E 写了 `Unif[0,2]`，但 Section 3.0.1 明确报告 mixed generation prob=0 或 0.2；代码按 Section 3 实现可控采样。 |
| token keep 概率用 \(\\kappa(\\min\\{1,\\tau_{text}\\})=\\kappa(t_{text})\) | `keep ~ Bernoulli(kappa(t_text))` | ✅ 对齐 | 代码额外 **强制保留 BOS**；若提供 `prompt_len` 则强制保留 prompt 前缀；若启用图像则强制保留 `<|oneflow_image|>` 占位符（便于按 interleaved schedule 决定是否删除）。 |
| 构造 noisy \(X_t\) 与每个 slot 的 bag-of-tokens \(\\mathcal{A}_i\) | `xt_list` / `bags_list` | ✅ 对齐 | bag 语义为“每个保留 token 后的插入槽位”，与论文 Algorithm 3 的 `A_j` 结构一致（依赖 BOS 被保留以保证第一个 bag 存在）。 |
| interleaved schedule：对每张图像采样 \(u\\sim\\text{Unif}(0,1)\)，\(\\tau_{img}=\\tau_{text}-\\kappa^{-1}(u)\)；若 \(\\tau_{img}<0\) 则图像 token 在该 snapshot 被删除并计入某个 bag；否则 \(t_{img}=\\min\\{1,\\tau_{img}\\}\) 并训练图像 flow matching | 实现了 `tau_img = tau_text - kappa_inverse(u)`，`tau_img<0` 删除 `<|oneflow_image|>` 并并入 bag；否则 `t_img=min(1,tau_img)` 并构造 `Y_t=tY1+(1-t)Y0` | ✅ 对齐 | 删除 token 时代码做了 bag 合并（删除一个“原本保留的 token”必须把其后 bag 合并进前一 bag），该处理是必要的。 |
| 图像 loss：Flow Matching \(\\|v(Y_t,t)-(Y_1-Y_0)\\|^2\) | `loss_img = mse(v, flow_tgt)`（可按 token 数归一） | ✅ 对齐 |  |
| 文本 loss：按 Eq. (7) 训练，包含：bag-of-tokens CE（Eq. 6）、zero-inflated 的 \(\\pi\) BCE（Eq. 5）、对非零缺失计数训练 \(\\lambda_{nonzero}\)（Eq. 4），并**明确不使用** \(\\dot\\kappa/(1-\\kappa)\) 对 loss 加权 | `text_loss_type=\"paper\"`（默认）实现 Eq (7)：token CE + BCE(pi) + Poisson(λ_nonzero,k>0)，且不对 loss 乘 w(t) | ✅ 对齐（默认） | 仍保留 `text_loss_type=\"ctmc\"` 作为 legacy 对照实现（会用 w(t) 加权）。 |
| 插入预测在实践中采用 **t-independent**（论文写明“不把 time 喂给网络预测 insertions”） | `condition_text_on_time=False`（默认）时，text token 的 `times` 置为常数（不随 `t_text` 变化）；图像 token 仍使用 `t_img` | ✅ 对齐（默认） | 若要回到“text 也条件于 time”，可设 `condition_text_on_time=True`。 |

---

## 1.1 小结（Trainer）

当前 `OneFlowTrainer` 默认配置（`text_loss_type=\"paper\"` + `condition_text_on_time=False`）已对齐论文的 **Algorithm 3** 训练目标（Eq. 7 + interleaved schedule + flow matching）。并额外实现了论文 Section 3.0.1 提到的 `mixed_generation_prob`（可设 0/0.2）。

如需做对照实验，仍可将 `text_loss_type` 设为 `"ctmc"` 以启用 legacy 的 CTMC-style loss（会用 \(\\dot\\kappa/(1-\\kappa)\) 对 loss 加权）。

---

## 2) Sampler（Algorithm 1-2）对齐矩阵

代码位置：`dllm/pipelines/oneflow/sampler.py`（`OneFlowSampler.sample`）

| 论文要求 | 代码现状 | 状态 | 备注 |
|---|---|---|---|
| Algorithm 1：初始化 \(X\\leftarrow\\emptyset\\)（空序列）、\(\\mathcal{I}\\leftarrow\\emptyset\\)、\(t_{text}=0\) | sampler 接受 `prompt` 输入；若输入为空则会至少塞一个 `BOS` | ✅/⚠️ | 对“无条件生成”严格来说应从空开始，但工程上通常需要 BOS 作为锚点；对“条件生成” prompt 固定更合理。 |
| Algorithm 2：每步先更新图像（Euler） \(Y\\leftarrow Y+\\Delta t_{img} v(Y,t_{img})\) | 对每个 image slice：`latent += dt_img * v[start:end]`，`t+=dt_img` | ✅ 对齐 |  |
| Algorithm 2：\(\\Delta t_{text}=\\min\\{1-t_{text},\\Delta t\\}\)，并行遍历所有插入槽位 | `dt_text = min(dt, 1-t_text)`，遍历每个 text token 的“其后槽位” | ✅ 对齐 | 遍历的是 token 索引 `i`，插入发生在 `i+1`，即 token 后的槽位。 |
| 若使用 \u03c0：\(p_i^{\\pi}=1-\\pi^i\)；否则跳过该 gate | `use_pi_gate` 分支：`p_pi = 1 - pi[pos]` 并 Bernoulli | ✅ 对齐 | sampler 默认 `use_pi_gate=True`。 |
| \(p_i^{\\lambda}=\\Delta t_{text}\\cdot \\frac{\\dot\\kappa(t_{text})}{1-\\kappa(t_{text})}\\cdot \\lambda^i_{nonzero}\) | `p_lam = dt_text * w(t_text) * lam[pos]` | ✅ 对齐 | 代码对 p 做了 clamp，避免 dt 过大时概率 >1。 |
| `do_insert = Bernoulli(p_pi) AND Bernoulli(p_lam)` | 先采 `do_lam`，再（可选）采 `do_pi`，两者均为真才插入 | ✅ 对齐 | 等价于 AND（独立采样）。 |
| 插入 token：\(a\\sim Q^i(\\cdot\\mid X)\)；若 \(a=<|image|>\) 则创建新图像 latent \(Y\\sim\\mathcal{N}(0,I)\)，并令 \(t_{img}(Y)=0\) | `a = sample_from_logits(q_logits[pos])`；若 `a==image_token_id` 则 `images.insert(..., latent=randn, t=0)` | ✅ 对齐 |  |
| 论文实践：插入预测采用 **t-independent**（不把 time 喂给网络预测 insertions） | `condition_text_on_time=False`（默认）时，sampler 让 text token 的 `times` 恒定；图像 token 仍用 `t_img` | ✅ 对齐（默认） | 可通过 `condition_text_on_time=True` 显式恢复 time-conditioned text。 |

---

## 3) Recipe 级对齐（paper-specified vs repo-actual）

这一节只对照论文中**明确写出的训练/数据设置**（Section 3.0.x / Appendix B.6 等），其余未写明项标为 *paper-unspecified*。

### 3.1 论文侧（可从文中直接摘到的）

- **训练阶段**：multimodal pretraining + instruction finetuning（Section 3.0.1）
- **预训练**：
  - **sequence length = 512**
  - **global batch size = 4096**
  - **mixed generation probability = 0 或 0.2**（“clean text 与 image 并发生成”的概率）
- **预训练数据**：filtered CC12M + YFCC + licensed data，总计 400M image-text pairs（Section 3.0.2）
- **k-scheduler**：\(\kappa_t=t^k\) 的 ablation 显示 **线性 k=1 最好**（Appendix B.6）

### 3.2 Repo 侧（`pt_wds_latents.py` 作为基线入口）

入口与配置：
- `examples/oneflow/pt_wds_latents.py`
- `dllm/utils/configs.py`（通用 TrainingArguments 默认值）
- `scripts/accelerate_configs/npu_ddp.yaml`（NPU 分布式配置）

核心可核对项（repo-actual）：
- **κ 调度**：默认 `scheduler_cls=LinearKappaScheduler` ✅ 与论文 B.6 一致
- **global batch（默认）**：
  - `per_device_train_batch_size=8`（脚本默认）
  - `gradient_accumulation_steps=1`（继承自 repo TrainingArguments 默认）
  - `npu_ddp.yaml` 默认 `num_processes=16`
  - ⇒ 默认 global batch = 8 × 1 × 16 = **128**（与论文 4096 不同，属于规模差异，不是算法错误）
- **sequence length（默认）**：
  - caption token 上限常见为 `max_caption_tokens=128`（数据预处理侧），并在 unified 序列中包含图像 latent tokens（数量由 latent H×W 决定，例如 16×16→256 tokens）
  - ⇒ 总长度与论文的 512 可能同量级，但并非严格对齐（可按需要配置/裁剪）
- **mixed generation probability（当前实现）**：
  - `trainer.py` 支持 `mixed_generation_prob=p`：以概率 `p` 采样 `τ_text∼Unif[1,2]`，否则 `τ_text∼Unif[0,1]`
  - 默认 `p=0.0`，可设为 `0.2` 对齐论文报告值 ✅

*paper-unspecified（论文未在 Section 3 明确写出，至少我们当前审计范围未看到的）*：
- optimizer 类型/β/ε/weight decay
- lr schedule 细节（cosine/linear 等）、warmup 配置
- grad clip、EMA、precision（bf16/fp16）等工程细节

