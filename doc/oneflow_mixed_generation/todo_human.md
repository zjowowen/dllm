# Mixed Generation 训练实验 TODO (Human Execution)

实验进展记录在 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md` 中。

---

## 准备工作

* 确认 Phase 1a（text-only CTMC 基线）和 Phase 1b（image-only overfit）均已通过
* 准备混合数据：WDS latent shards（flower32）+ 文本数据（fineweb-edu）
* 确认 910C 环境可用

---

## 实验 2a-1：基线（CTMC + prob=0.5 + weight=1.0）

**目标**：验证 text + image 能同时优化。

**说明**：当前 worktree 未提供 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_mixed_generation/launch_pt_mixed_910c.sh`；请改用现有训练入口 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_mixed_generation/pt_mixed.py`。当前也未提供 dedicated mixed-generation eval 脚本。

**执行命令**：
```bash
accelerate launch \
  --config_file /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_mixed_generation/pt_mixed.py \
  --tokenizer_name_or_path /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer \
  --shards "/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32" \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/mixed_gen_2a1_baseline \
  --max_steps 2000 \
  --mixed_generation_prob 0.5 \
  --image_loss_weight 1.0 \
  --text_loss_type ctmc \
  --condition_text_on_time True \
  --log_split_losses True \
  --save_steps 500 \
  --logging_steps 10
```

**验收标准**：
- loss_text 和 loss_img 均下降
- 无 NaN / 梯度爆炸
- 两种 loss 不互相"拉扯"

---

## 实验 2a-2：高图像权重（weight=5.0）

**执行命令**：
```bash
accelerate launch \
  --config_file /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_mixed_generation/pt_mixed.py \
  --tokenizer_name_or_path /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer \
  --shards "/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32" \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/mixed_gen_2a2_w5 \
  --max_steps 2000 \
  --mixed_generation_prob 0.5 \
  --image_loss_weight 5.0 \
  --text_loss_type ctmc \
  --condition_text_on_time True \
  --log_split_losses True \
  --save_steps 500 \
  --logging_steps 10
```

**分析目标**：对比 weight=1.0 vs weight=5.0 对 text loss 和 image loss 各自的影响。

---

## 实验 2a-3：低混合比例（prob=0.2）

**执行命令**：
```bash
accelerate launch \
  --config_file /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_mixed_generation/pt_mixed.py \
  --tokenizer_name_or_path /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer \
  --shards "/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32" \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/mixed_gen_2a3_p02 \
  --max_steps 2000 \
  --mixed_generation_prob 0.2 \
  --image_loss_weight 1.0 \
  --text_loss_type ctmc \
  --condition_text_on_time True \
  --log_split_losses True \
  --save_steps 500 \
  --logging_steps 10
```

**分析目标**：低图像比例下 image loss 是否仍能有效优化。

---

## 实验 2a-4：Paper Eq7 对照

**执行命令**：
```bash
accelerate launch \
  --config_file /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_mixed_generation/pt_mixed.py \
  --tokenizer_name_or_path /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer \
  --shards "/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32" \
  --output_dir /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts/mixed_gen_2a4_paper \
  --max_steps 2000 \
  --mixed_generation_prob 0.5 \
  --image_loss_weight 1.0 \
  --text_loss_type paper \
  --condition_text_on_time False \
  --log_split_losses True \
  --save_steps 500 \
  --logging_steps 10
```

**分析目标**：Paper Eq7 的不稳定性是否在多模态场景下更严重。

---

## 执行后操作

1. 将每个实验的 loss 曲线（text 和 image 分开）截图更新到 `PROGRESS.md`
2. 记录最终 loss 数值和梯度范数范围
3. 确定最优 mixed_generation_prob 和 image_loss_weight 组合
