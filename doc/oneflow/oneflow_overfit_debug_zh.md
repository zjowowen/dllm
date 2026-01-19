# OneFlow 过拟合（Overfit）Debug 手册：训练 vs 采样 vs 不匹配（中文）

本手册面向这样一个非常具体的目标：**在极小数据集（例如 32 条样本）上“过拟合成功”**，并判定失败原因究竟来自：

- **训练问题**：训练没把 image loss 真正优化起来（或优化很弱/不稳定）
- **评估/采样问题**：采样配置不对、prompt 不对、推断走了和训练不一致的路径
- **训练-评估不匹配**：token/latent 约定（BOS/EOS、latent flatten、latent scale、image token 对齐等）不一致
- **数据集本身不可用**：latents 解码失败/解码出来不是你以为的内容（caption 里含 flower 但图像不是花很常见）

> 你的 overfit 判据（最强、也最容易暴露问题）：
> **用训练集中的 caption 作为 prompt，生成的图像应尽量接近该样本的 GT 图像**（可用肉眼 + 简单指标如 latent MSE / pixel PSNR）。
---

## 1. 两个核心脚本：训练与采样到底做了什么

### 1.1 训练入口：`examples/oneflow/pt_wds_latents.py`

#### 输入数据（WebDataset tar）约定
每条样本（tar 内一个 key）应包含：
- `npy`：VAE latent，常见 shape 为 `[4, 16, 16]`（128×128 图像对应 16×16 latent 网格），**已乘 latent_scale（默认 0.18215）**。
- `txt`：caption 文本
- `json`（可选但你当前使用的是必须的）：`{"input_ids": [...]}`，其中 `input_ids` 已经包含：
  - `[BOS] + caption_ids + [<|oneflow_image|>] + [EOS]`

这些由 `scripts/oneflow/precompute_latents_wds.py` 生成。

#### Dataset 如何产出 feature
`WDSLatentsIterableDataset._to_feature()` 最终返回：
- `input_ids`: `List[int]`（来自 `json.input_ids`，或即时 tokenize）
- `image_latent`: `Tensor`（来自 `npy`，shape 可能是 `[4,H,W]` 或 `[N,4]` 等）

#### Collator 如何进入 Trainer
`dllm/pipelines/oneflow/utils.py::OneFlowCollator` 会把 batch 组装为：
- `x1_ids`: `List[List[int]]`（保留可变长 list；这点非常关键）
- `image_latents`: `List[Tensor|List[Tensor]|None]`（这里每条样本一般是单张图，所以是 `Tensor`）

Trainer 侧的 `compute_loss` 强依赖 `x1_ids` 和 `image_latents`：
- 如果 `image_latents` 全为 None，则走 **text-only** 路径，**image loss 恒为 0**。
- 如果 `image_latents` 非空，才会走 **mixed-modal** 路径并计算 image flow matching loss。

#### Mixed-modal loss（核心）
`dllm/pipelines/oneflow/trainer.py::OneFlowTrainer.compute_loss` 在 mixed-modal 路径中，对每个被保留的 `<|oneflow_image|>`：
1. 取 GT latent `y1`（来自数据集，已缩放）
2. 采样 `y0 ~ N(0, I)`（同 shape）
3. 采样 `t_img`（由 interleaved schedule 得到）
4. 构造：
   - \(y_t = t_{img} y_1 + (1-t_{img}) y_0\)
   - flow target：\(v^* = y_1 - y_0\)
5. 将 `y_t` **展平为 `[N, 4]`** 并插入到 unified sequence（位于 `<|oneflow_image|>` token 之后）
6. 模型输出 `v`（同位置、同 shape），image loss 为：
   - \(\|v - v^*\|_2^2\) 在所有 modality token 上求和/求均值

**重要不变量（训练侧）**：
- `<|oneflow_image|>` token 的出现次数必须与 `image_latents` 数量一致，否则 Trainer 会直接报错。
- `flatten_latent` 支持的输入形状：`[N,4]` / `[4,H,W]` / `[H,W,4]`，最终统一成 `[N,4]`（row-major，H→W）。

---

### 1.2 采样入口：`examples/oneflow/sample_and_decode.py`

该脚本做两件事：
1) 用 `OneFlowSampler` 从 checkpoint 采样，得到 `out.images`（list，元素为 `[N,4]` latent tokens，**已处于缩放 latent 空间**）
2) 用 VAE decode 成 PNG：
   - reshape：`[N,4] -> [1,4,H,W]`
   - 反缩放：`lat /= latent_scale`
   - `vae.decode(lat)` 得到像素

#### `OneFlowSampler` 的关键点
`dllm/pipelines/oneflow/sampler.py::OneFlowSampler.sample`（v1）：
- **只支持 bs=1**
- 会强制在 prompt 最前面补 `BOS`（若你传入的 token 列表不以 BOS 开头）
- 若 prompt 里包含 `<|oneflow_image|>`，会为每个 image token 初始化一个：
  - `latent ~ N(0,I)`，shape `[image_num_tokens, 4]`
  - `t=0`
- 每一步：
  - 构造 unified sequence：把 image token 后面插入 N 个 modality tokens（dummy ids + modality_tokens）
  - 前向得到 `v`
  - 对每张图做 Euler 更新：`y += dt * v`，直到 `t_img` 走到 1

**重要不变量（采样侧）**：
- `image_num_tokens` 必须等于 `latent_h * latent_w`（128×128 → 16×16 → 256）。
- 输出的 latent tokens 与训练的 flatten 约定一致，才能正确 reshape/decode。

---

## 2. 训练 vs 采样“最常见的不匹配点”

### 2.1 Prompt tokenization 不一致（推荐用训练集的 input_ids 做评估）
训练时，如果你使用 `use_precomputed_ids=True`，则训练的 `x1_ids` 是固定格式：
- `[BOS] + caption + [<|oneflow_image|>] + [EOS]`

但 `sample_and_decode.py` 默认使用：
- `tokenizer.encode(prompt, add_special_tokens=False)`
- 然后 `OneFlowSampler` 自动补 BOS
- **不会自动补 EOS**

这通常不会“完全错误”，但如果你要做 **严格 overfit / reconstruction**，建议评估时直接用训练样本自带的 `json.input_ids`，这样训练-评估 token 序列完全一致。
> 后续我们会提供一个 overfit eval 脚本：从 WDS 取一条训练样本 → 用其 `input_ids` 采样 → 与 GT 解码对比。

### 2.2 数据集 caption 过滤 ≠ 图像内容真的像花
用 `caption_regex=\\bflower\\b` 抽 32 条样本，只保证 caption 里出现了 “flower” 单词：
- 可能是“花在头发上/衣服花纹/节日卖花”等，图像主体不一定是花
- 甚至可能是 URL 对应图片与 caption 弱相关（CC3M 常见）

所以必须先做 **GT latents 解码验收**（Gate A，见下文）。

### 2.3 过拟合目标的“理论预期”
你希望“caption → 复现该样本的 GT 图像”，这本质上是 **把生成模型当成确定性条件重建模型**。
OneFlow 的训练目标更像“学习条件分布”，并不保证一一映射；但在极小数据集上通常仍会出现强记忆。
因此我们推荐用 **GT-vs-Gen 的指标随训练步数是否下降** 来判断过拟合趋势，而不是只看单张图像像不像花。

---

## 3. Debug Gate（一步步判定是训练还是评估的问题）

下面每个 Gate 都是“可执行 + 明确 pass/fail”。
只要按顺序做，就能快速定位问题归因。

### Gate A：数据集本身能否 decode 成功？（必须先过）
目标：确认 WDS 里的 `npy` latents 能解码成合理图像，并且确实与 caption 大致一致。

做法：使用脚本：
- `scripts/oneflow/verify_wds_latents_decode.py`

建议先解码 8–32 条样本，人工看一眼是否真的是“花”。
若 Gate A 不过：不要继续调训练/采样，先修数据集抽样策略。

### Gate B：训练是否真的在优化 image loss？
目标：确认 `compute_loss` 走的是 mixed-modal 路径，并且 `loss_img` 非 0（至少在部分 step）。

做法：
- 开启训练侧 debug log（默认关闭；训练命令加 `--debug_log_first_batch True`）
- 观察 rank0 的一次性输出：`has_images`, `img_tokens`, `loss_img`, `loss_text`

若 Gate B 不过：通常是 `image_latents` 没正确进入 batch（collator/key 不匹配）或数据集缺失 `npy`。

### Gate C：用训练样本做 reconstruction，对比 GT-vs-Gen 是否随 step 改善？
目标：固定一个训练样本（caption + latent），对多个 checkpoint 执行：
1) decode GT
2) 采样生成图
3) 计算 metrics（latent MSE / pixel PSNR）
并观察指标随训练步数是否下降/上升。

做法：使用脚本：
- `examples/oneflow/overfit_eval_wds.py`

若 Gate A & B 都过，但 Gate C 完全无改善：
- 优先怀疑 **采样配置**（dt 太大/步数太少/温度等）或 prompt 不一致
- 其次怀疑 **训练过强删图**（极端情况下 batch 里图片都被 τ_img delete，导致 image loss 很弱）
---

## 4. 推荐的“最硬核 overfit”设置（更容易看到趋势）

### 4.1 数据集选择建议
为了让 reconstruction 更可能成功，建议做两层：
1) 先用 **1 张样本** 过拟合（最容易看到记忆）
2) 再扩到 8/32 张

### 4.2 采样建议
如果生成不稳定，尝试更小 dt 和更大步数（Euler 更稳定）：
- `dt=0.02, max_steps=200`
并固定随机种子，便于对比 checkpoint 变化。

---

## 5. 你现在最该做的三件事（执行顺序）
1) **跑 Gate A**：把 `wds_latents_flower32` 的 GT latents 解码成 PNG，确认“真的是花/至少像花”。
2) **跑 Gate C**：固定一个训练样本 key，用多个 checkpoint 对比 GT-vs-Gen 指标。
3) 若指标在变好但肉眼还是不像花：说明 dataset 本身不够“花”；换更干净的子集（或改过滤条件）。

