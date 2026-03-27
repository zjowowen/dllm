# Interleaved Generation 实验 TODO (Human Execution)

实验进展记录在 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md` 中。

---

## 准备工作

* 确认 Phase 2a（mixed generation 基线）已完成，确定最优 image_loss_weight
* 准备数据：WDS latent shards + 文本数据
* 确认 910C 环境可用
* 确认 VAE decoder 可用（用于 Phase 3 采样评估）

---

## Phase 2b 实验

### Exp 2b-1：CTMC 基线（全 Interleaved）

**目标**：验证完整 τ_text ~ Unif[0,2] interleaved schedule。

**说明**：当前 worktree 未提供 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_interleaved/launch_pt_interleaved_910c.sh`；请改用现有训练入口 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/pt_interleaved.py`。当前也未提供 dedicated interleaved loss / sample eval 脚本。

**执行命令**：
```bash
accelerate launch \
  --config_file /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/pt_interleaved.py \
  --tokenizer_name_or_path /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer \
  --shards "/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32" \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/interleaved_2b1_baseline \
  --max_steps 5000 \
  --tau_text_max 2.0 \
  --image_loss_weight 1.0 \
  --text_loss_type ctmc \
  --condition_text_on_time True \
  --log_split_losses True \
  --save_steps 500 \
  --logging_steps 10
```

**验收标准**：
- Loss 无异常（特别关注 τ_text ∈ (0.9, 1.1) 过渡区间）
- 图像删除率统计合理
- text_loss 和 image_loss 均可下降

---

### Exp 2b-3：Paper Eq7 对照

**执行命令**：
```bash
accelerate launch \
  --config_file /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/pt_interleaved.py \
  --tokenizer_name_or_path /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer \
  --shards "/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32" \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/interleaved_2b3_paper \
  --max_steps 5000 \
  --tau_text_max 2.0 \
  --image_loss_weight 1.0 \
  --text_loss_type paper \
  --condition_text_on_time False \
  --log_split_losses True \
  --save_steps 500 \
  --logging_steps 10
```

---

## Phase 3 实验

### Exp 3a-1：Text-only 采样

```bash
python -u /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
  --mode text_only \
  --prompt "OneFlow is a generative model" \
  --max_new_tokens 128 \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/outputs/interleaved_3a1_text
```

### Exp 3b-1：Image-conditioned 采样

```bash
python -u /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
  --mode image_conditioned \
  --prompt "a photo of a flower" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --max_steps 40 --dt 0.05 \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/outputs/interleaved_3b1_image
```

### Exp 3c-1：Full Interleaved 采样

```bash
python -u /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
  --mode interleaved \
  --prompt "a photo of a flower" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --max_steps 40 --dt 0.05 \
  --max_new_tokens 256 --max_seq_len 512 \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/outputs/interleaved_3c1_full
```

**验收标准**：
- 完整生成链路不崩溃
- 文本有基本连贯性
- 图像 VAE decode 后非纯噪声
- 终止条件正常触发

---

## 一致性检查

### 统一序列构建 Diff

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm
pytest -q /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/tests/test_oneflow_sequence_ops.py /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/tests/test_oneflow_sampler_step.py /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/tests/test_oneflow_interleaved_schedule.py
```

---

## 执行后操作

1. 将 Phase 2b 的 loss 曲线（分 text/image）截图更新到 `PROGRESS.md`
2. 将 Phase 3 的采样结果（文本 + 图像截图）更新到 `PROGRESS.md`
3. 记录图像删除率统计
4. 确定是否需要进入 Phase 4 消融
