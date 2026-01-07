# OneFlow 多模态（Text Edit Flow + Image Flow Matching）设计与落地方案（中文）

> 目标：把 OneFlow（插入式 Edit Flow 文本 + Flow Matching 图像 latent + 交错调度 κ）的算法完整加入本仓库，并与现有 `dllm/pipelines/editflow` 的工程结构保持一致。
>
> 参考：
> - OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows — [arXiv:2510.03506](https://arxiv.org/html/2510.03506)
> - Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model — `https://arxiv.org/html/2408.11039`
> - Edit Flows（前作，仅用于工程风格与 CTMC 损失对齐）：`https://arxiv.org/html/2506.09018`

---

## 总览

- 本仓库当前重点是“扩散式文本模型（dLLM）”与其训练/推断脚手架；`editflow` 提供了 CTMC 风格的 edit operation 训练器与采样器。
- OneFlow 的关键变化是 **多模态**：文本用插入式 edit flow，图像用 latent-space flow matching，并通过 **交错时间表 κ** 让两种模态并行生成。
- 由于你已经提供了 `reference/transfusion-pytorch`，图像 latent 与 unified trunk 将以其为蓝本（并会在 `dllm` 内部 vendoring / internal module 化）。

---

## 1. 设计目标与非目标

### 目标
- 新增 `dllm/pipelines/oneflow/`：包含模型、训练器、采样器、collator、工具函数。
- 训练侧实现 OneFlow Algorithm 3（interleaved training）：
  - 文本：删除噪声（κ-keep）→ bag-of-tokens 监督 → 插入强度 λ 与 token 分布 Q。
  - 图像：按 `τ_img = τ_text - κ^{-1}(u)` 决定是否“已插入”，并对已插入的 latent 做 flow matching。
- 推断侧实现 OneFlow Algorithm 1-2（interleaved generation）：
  - 文本：并行插入（Bernoulli 触发）；
  - 图像：Euler 更新 latent（或 ODE solver 可选）。
- `examples/oneflow/` 提供最小可跑脚本（text-only / mixed-modal / sample / chat）。
- 文档与依赖：在 `doc/` 写清楚数据格式、训练阶段、超参建议、依赖安装与 license 说明。

### 非目标（第一版）
- 不追求复刻 OneFlow 论文的全部训练 recipe/数据配方/模型规模；第一版以算法正确性与工程可扩展为主。
- 不强制集成某个特定大模型（如 LLaMA init）；第一版先支持“从零搭建 unified trunk + heads”，后续可扩展到迁移学习/初始化。

---

## 2. 数据与表示

## 2.1 Ground-truth 样本结构

OneFlow 的核心训练输入是一条“交错序列”，包含：
- **文本 token**：离散序列 \(X\)
- **图像 latent**：连续张量 \(Y_1\)（每张图像对应一个 latent 张量）
- **图像在序列中的位置**：需要能把每个图像绑定到某个“插入槽位（gap）”，以便在 `τ_img<0` 时把 `<|oneflow_image|>` 放回正确的 bag-of-tokens `𝒜_j`。

推荐的 repo 内部表示（借鉴 transfusion 的 `ModalitySample`，但更贴合 OneFlow）：
- 每条样本表示为一个 Python list（“段落列表”）：
  - 文本段：`torch.LongTensor[seq]`（token ids）
  - 图像段：`torch.FloatTensor[...]`（latent；可为 `(C,H,W)` 或 `(H,W,C)` 或 flattened `(N,d)`，由 config 决定）
- 文本段中使用 special token `<|oneflow_image|>` 作为“图像占位符”（即论文里的 `<|image|>`），并在段落列表中紧跟对应的图像 latent 段。这样可以在训练/推断时自然对齐“图像出现的位置”。

## 2.2 Collator 输出

`OneFlowCollator`（位于 `dllm/pipelines/oneflow/utils.py`）将把 batch 的样本整理成训练器可用的结构，至少包含：
- `x1_ids`: `List[List[int]]`（ground-truth token ids；包含 BOS/EOS 与 `<|oneflow_image|>`）
- `image_latents`（可选）：预编码 latents，并与 `x1_ids` 中 `<|oneflow_image|>` 的出现顺序对齐
- `prompt_len`（可选）：用于 SFT 时屏蔽 prompt loss / 或限制采样时编辑 prompt

---

## 3. 模型架构

## 3.1 总体结构

`OneFlowModel`（`dllm/pipelines/oneflow/models/oneflow_model.py`）分为三块：
- **Unified trunk（Transfusion 风格 Transformer）**：处理混合 token（文本 embedding + 图像 latent projection），并接收 per-token time conditioning。
- **文本插入头（Text insertion heads）**：输出 OneFlow 文本侧所需的 `{π, λ_nonzero, Q}`。
- **图像 flow 头（Image flow head）**：输出图像 latent token 的 velocity \(v(Y_t, t)\)（或等价 flow）。

## 3.2 Unified trunk：来自 Transfusion 的关键点

我们会把 `reference/transfusion-pytorch` 中与 trunk 相关的模块 vendoring 到 `dllm`（保留 license 注记），主要使用其：
- time conditioning（Random Fourier embedding + AdaLN / AdaptiveWrapper）
- rotary embedding 与“relative positions derived from modality positions”
- mixed attention mask：文本默认双向，图像块内允许双向（或特殊规则），并支持 `modality_positions` 驱动的 mask

注意：Transfusion 原始实现同时支持 AR 文本 next-token loss。OneFlow 文本侧不是 AR，因此我们只复用其 trunk/utility，并在上层实现 OneFlow 的插入头与损失。

## 3.3 文本插入头

- 插入槽位（slot）定义：对当前文本序列 \(X_t\) 的每个 token 位置 \(i\)（包含 BOS，通常不包含 EOS），定义“在其后插入”的槽位。
- 头的输出：
  - `pi`: `sigmoid(W_pi h_i)` → \(π^i\)
  - `lambda_nonzero`: `softplus(W_lam h_i)` → \(λ_{\text{nonzero}}^i\)
  - `Q_logits`: `W_q h_i` → softmax 得到 \(Q^i(\cdot)\)
- `Q` 的词表应包含：
  - 常规文本 token
  - `<|oneflow_image|>`（触发生成一张图像）
  -（可选）`<|oneflow_image_som|>`, `<|oneflow_image_eom|>` 等调试 token

## 3.4 图像 head（flow / velocity）

对属于图像 latent 的 token 位置，输出一个与 latent 维度相同的向量：
- `v = W_img h`（可为单层 linear，或小 MLP）
用于 flow matching：监督目标为 \(Y_1 - Y_0\)（直线流）。

---

## 4. 训练方案（Algorithm 3）

## 4.1 κ 调度与 κ^{-1}

本仓库已有 `dllm/core/schedulers/kappa.py` 提供：
- `kappa(t)`
- `kappa_derivative(t)`
- `weight(t) = κ'(t)/(1-κ(t))`

OneFlow 还需要 `κ^{-1}(u)`（论文 Eq. 28）。实现策略：
- Linear：\(κ(t)=t\) → \(κ^{-1}(u)=u\)
- Cosine：\(κ(t)=1-\cos(\pi t/2)\) → \(κ^{-1}(u)=\frac{2}{\pi}\arccos(1-u)\)
- 其他：使用数值二分（u∈[0,1]，t∈[0,1]）

## 4.2 文本侧 noising 与 bag-of-tokens

（对应论文 Algorithm 3 第 5-14 行）

给定 ground-truth token 序列 \(X\)：
- 采样 `τ_text ~ Unif[0,2]`，设 `t_text = min(1, τ_text)`。
- 对每个 token 做 κ-keep：保留概率 `κ(t_text)`；BOS 强制保留。
- 构造：
  - `X_t`: 保留 token 形成的子序列（保持顺序）
  - `A_j`: bag-of-tokens（每个插入槽位的“缺失 token 多重集合”）

bag 的构造与论文一致（伪码）：
- 设 `j=0`
- 遍历 `x in X`：
  - 若保留：`X_t.append(x)`，`j += 1`，并初始化新 `A_j = []`
  - 若删除：将 `x` 追加到当前 `A_j`

直观理解：`A_j` 收集的是“第 j 个保留 token 之后缺失的 token”。如果 BOS 总是保留，那么 `A_1` 就对应 BOS 后的插入槽位。

## 4.3 文本损失（insertion edit flow）

在时间 `t_text`，记 `w = κ'(t_text)/(1-κ(t_text))`。模型在每个槽位 i 输出 `λ_i` 与 `Q_i`。

推荐实现为 CTMC NLL 的 MC estimator（与 `editflow` 的 survival + positive term 风格对齐）：
- **Survival term**：
  - \(L_{\text{surv}} = \mathbb{E}[ w \sum_i λ_i ]\)
- **Positive term**：
  - 对每个槽位 i 的 bag `A_i`，若其中有 k 个 token \(a_1..a_k\)，则
  - \(L_{\text{pos}} = \mathbb{E}[ w \sum_i \sum_{a\in A_i} (-\log λ_i - \log Q_i(a)) ]\)
- **π term（可选）**：若启用 π 门控，则可加一个辅助 BCE；或仅在采样时使用而不训练。

实践细节：
- `λ_i` 用 `softplus` 保证非负，并对数值稳定做 clamp。
- `Q_i` 用 `log_softmax`，对目标 token 做 NLL。
- 可按“原始长度/有效槽位数”归一化，使 batch 间尺度稳定。

## 4.4 图像侧 interleaved 时间与 flow matching

（对应论文 Algorithm 3 第 16-28 行）

对每张图像（ground-truth latent 记为 `Y1`）：
- 采样 `u~Unif(0,1)`，计算 `τ_img = τ_text - κ^{-1}(u)`
- 若 `τ_img < 0`：
  - 当前 snapshot 视为“图像尚未插入”，等价于 `<|oneflow_image|>` 这个 token 在文本序列里被删除
  - 因此需要把 `<|oneflow_image|>` 加入对应槽位的 bag `A_j`
- 否则：
  - `t_img = clip(τ_img)`（限制到 [0,1]）
  - 采样 `Y0 ~ N(0,I)`
  - 构造 `Y_t = t_img * Y1 + (1 - t_img) * Y0`
  - flow target 为 `flow = Y1 - Y0`
  - 最终 `L_image = MSE(vθ(Y_t, t_img), flow)`

## 4.5 总损失与权重

- `L = L_text + w_img * L_image (+ optional terms)`
- 建议默认：`w_img=1.0`，并按图像 token 数归一图像 loss（防止图像 token 过多导致梯度主导）。

## 4.6 训练阶段建议

- Stage A：Text-only 插入流（先把 λ/Q 学稳定）
- Stage B：Image-only latent flow matching（可条件于 caption 或无条件）
- Stage C：Mixed-modal interleaved（OneFlow 核心）
- Stage D：多模态 SFT（对话/VQA）

---

## 5. 推断方案（Algorithm 1-2）

## 5.1 状态变量

采样器维护：
- `X`: 当前文本 token 序列（动态长度）
- `I`: 图像 latent 集合，每个元素带 `t_img`（时间）
- `t_text`: 文本时间（0→1）

## 5.2 单步更新

（对应 Algorithm 2）

每一步 `Δt`：
1) `model(X, I, t_img)` 前向，得到 `{π, λ, Q}` 与各图像的 `v(Y,·)`；
2) **图像**：对所有 `t_img<1` 的图像做 Euler：
   - `Δt_img = min(1 - t_img, Δt)`
   - `Y ← Y + Δt_img * v(Y, t_img)`
   - `t_img ← t_img + Δt_img`
3) **文本**：`Δt_text = min(1 - t_text, Δt)`，并行遍历所有 insertion slot：
   - `p_i^λ = Δt_text * κ'(t_text)/(1-κ(t_text)) * λ_i`
   - 若启用 π：`p_i^π = 1 - π_i`；`do_insert = Bernoulli(p_i^λ) & Bernoulli(p_i^π)`
   - 若插入：`a ~ Q_i(·)`，执行 `X = ins(X, i, a)`
   - 若 `a == <|oneflow_image|>`：创建新图像 latent `Y~N(0,I)`，`t_img=0`，并插入到 `I`
4) `t_text ← t_text + Δt_text`

终止：
- `t_text>=1` 且所有图像 `t_img>=1`

## 5.3 解码

若训练/推断中使用 VAE：
- 采样结束后把每个图像 latent `Y` 送入 `VAEDec` 得到像素图像。
否则：
- 直接返回 latent（便于下游或单元测试）。

---

## 6. 数据集推荐与数据格式

## 6.1 推荐数据集

- 文本预训练：FineWeb / C4 / The Pile / OpenWebText（toy: tiny-shakespeare）
- 图像-文本预训练：LAION 子集、CC3M/CC12M、COCO Captions、Flickr30k、DataComp
- 多模态指令：VQAv2、OK-VQA、GQA、LLaVA 系列、ShareGPT4V

## 6.2 最小数据格式（建议）

为了让第一版可跑，建议先支持最简单的 `(caption, image)`：
- `text`: str（caption）
- `image`: PIL.Image 或 numpy/torch tensor（RGB）

通过 map_fn 把它转成：
- 文本 token：`[BOS] + tokenize(text) + [<|oneflow_image|>] + [EOS]`
- 图像：编码到 latent `Y1`（在线或离线）

---

## 7. 仓库落地与接口

### 新增路径
- `dllm/pipelines/oneflow/`
- `examples/oneflow/`
- `doc/oneflow/oneflow_design_zh_en.md`（本文件）

### 与现有风格对齐
- 训练脚本保持与 `examples/editflow/pt.py` / `sft.py` 相同结构：`HfArgumentParser` + `dllm.utils.initial_training_setup` + 自定义 trainer。
- 推断脚本保持与 `examples/editflow/sample.py` / `chat.py` 相同结构：构建 sampler → `sample()` → 可视化/解码。

### 与 editflow 的 utils 复用与共享模块

为减少 `editflow` 与 `oneflow` 在 CTMC/采样工具函数上的重复实现，并避免 pipeline 间相互依赖造成的循环 import，本仓库把“通用 helper”抽到共享模块：

- `dllm/pipelines/ctmc_utils.py`
  - `pad_1d`: 训练侧把变长 token list pad 成 `[B,L]` 张量与 mask
  - `sample_from_logits`: 从 logits 采样 token（支持 temperature）
  - `bernoulli_from_rate`: 把 rate 与步长 τ 转为 Bernoulli 触发（带 clamp）
  - `safe_log`: 数值稳定的 `log(x)`（带 clamp）

迁移后的依赖关系：
- `dllm/pipelines/editflow/{trainer.py,sampler.py}` 与 `dllm/pipelines/oneflow/{trainer.py,sampler.py}` 统一 import `ctmc_utils`。
- 为保持兼容，`dllm/pipelines/editflow/utils.py` 继续对外暴露 `pad_1d`（通过 re-export），旧的 import 路径不会被破坏。

### prompt_len 语义（SFT）

当 batch 中提供 `prompt_len`（例如 SFT 场景 `prompt + response`）时，OneFlow v1 采用与 `EditFlow` 一致的语义：**prompt 前缀只作为条件输入，不在训练中被编辑**。

- **训练（OneFlowTrainer）**：
  - `prompt_len` 表示 `x1_ids` 的前缀长度（包含 BOS 时也应计入）。
  - 在构造 `X_t` 时，trainer 会强制 keep `x1_ids[:prompt_len]`，从而避免 prompt 区域产生删除/插入事件。
  - 约束：`0 < prompt_len <= len(x1_ids)`，否则报错。
- **推断（OneFlowSampler）**：
  - 默认 `edit_prompt=False`，只允许在 prompt 的最后一个 token 之后插入（保持前缀稳定）。
- **多模态限制（v1）**：
  - 若训练 batch 同时提供 `image_latents`，并且 `prompt_len` 覆盖了 `<|oneflow_image|>`，当前实现会显式报错。
  - 原因：v1 尚未定义“prompt 内图像作为条件输入”的语义（避免把条件图像误当作待生成图像）。
  - 后续若要支持条件图像，建议引入独立 token/字段区分「条件图像」与「待生成图像」。

---

## 8. 依赖与 License

- Transfusion 参考实现需要一组额外依赖（见 `reference/transfusion-pytorch/pyproject.toml`），建议在本仓库 `pyproject.toml` 中新增 `oneflow` optional extra（避免影响默认安装）。
- vendoring 时必须保留上游 license（MIT）与出处说明（文件头注释或 `THIRD_PARTY_NOTICES`）。

---

## 9. 里程碑与验收

- 能跑通 text-only toy：插入式生成可增长序列。
- 能跑通 text+image toy：训练后能采样出至少一张图像 latent，并能（可选）通过 VAE 解码到像素图。
- 代码结构清晰、可扩展（未来可替换 trunk、支持多图、多模态、多分辨率）。


