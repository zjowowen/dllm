# Image-Only Flow Matching 实验 TODO (Human Execution)

实验进展记录在 `doc/oneflow_image_only/PROGRESS.md` 中。

---

## 准备工作

* 确认 WDS latent shards 可用（已在 Stage 0 验证）
* 确认 VAE decoder 可用（本地路径或 HF 缓存）
* 确认 910C 环境已激活（Ascend + torch_npu）

---

## 实验 1b-1：单 shard Overfit（核心验证）

**目标**：验证图像 flow matching 在 Transfusion trunk 上能正确优化。

**执行命令**：
```bash
bash scripts/oneflow_image_only/launch_pt_image_910c.sh \
  --shards "data/latents_128_bundle/wds_latents_flower32/shard-000005.tar" \
  --output_dir data/ckpts/image_only_overfit_1b1 \
  --max_steps 5000 \
  --per_device_train_batch_size 1 \
  --image_loss_weight 1.0 \
  --save_steps 500 \
  --logging_steps 10
```

**验收标准**：
- image_loss 从初始值持续下降，5000 步后明显低于初始值
- 无 NaN / Inf
- 采样 + VAE decode 后的图像非纯噪声

---

## 实验 1b-2：image_loss_weight 对比

**目标**：确定 image_loss_weight 对训练速度和质量的影响。

**执行命令**：
```bash
# weight = 1.0 (同 1b-1)
# weight = 10.0
bash scripts/oneflow_image_only/launch_pt_image_910c.sh \
  --shards "data/latents_128_bundle/wds_latents_flower32/shard-000005.tar" \
  --output_dir data/ckpts/image_only_overfit_1b2_w10 \
  --max_steps 5000 \
  --per_device_train_batch_size 1 \
  --image_loss_weight 10.0 \
  --save_steps 500 \
  --logging_steps 10
```

**分析目标**：对比 weight=1.0 vs weight=10.0 的 loss 下降速度和最终重建质量。

---

## 实验 1b-3：全量 flower32

**目标**：多样本下 loss 趋势。

**执行命令**：
```bash
bash scripts/oneflow_image_only/launch_pt_image_910c.sh \
  --shards "data/latents_128_bundle/wds_latents_flower32" \
  --output_dir data/ckpts/image_only_flower32_1b3 \
  --max_steps 1000 \
  --per_device_train_batch_size 4 \
  --image_loss_weight 1.0 \
  --save_steps 200 \
  --logging_steps 10
```

**分析目标**：多样本下 image loss 是否仍持续下降（非 overfit，泛化趋势）。

---

## 采样评估

### 批量采样（推荐）

在每个实验的最优 checkpoint 上运行批量图像采样评估：
```bash
python -u scripts/oneflow_image_only/eval_image_sample.py \
  --model_dir <checkpoint_dir> \
  --prompts_file scripts/oneflow/eval_prompts_image_v1.jsonl \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir data/vis/image_only_eval_<exp>
```

输出：`images/*.png`（每个提示一张图）+ `report.jsonl` + `report.md`。

### 单条采样

```bash
python -u examples/oneflow/sample_and_decode.py \
  --model_dir <checkpoint_dir> \
  --prompt "a photo of a flower <|oneflow_image|>" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir data/vis/image_only_samples \
  --dt 0.05 --max_steps 40 --image_num_tokens 256
```

### 过拟合评估（GT vs Gen）

```bash
python -u examples/oneflow/overfit_eval_wds.py \
  --model_dir <checkpoint_dir> \
  --wds_shards data/latents_128_bundle/wds_latents_flower32 \
  --sample_index 0 \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir data/vis/overfit_eval_<exp>
```

---

## 执行后操作

请将 loss 曲线截图或关键数据更新到 `PROGRESS.md`。
