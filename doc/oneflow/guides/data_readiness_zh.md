# OneFlow 实验数据就绪性审查（中文）

本文档审查 OneFlow 各实验阶段的训练数据和验证数据就绪情况，并提供缺失数据的准备流程。

**最后更新**: 2026-02

---

## 1. 数据就绪性总览

### 1.1 训练数据状态

| 阶段 | 数据类型 | 格式 | 当前路径 | 规模 | 分片数 | 状态 |
|---|---|---|---|---|---|---|
| **1a text-only** | 文本 PT | HF Arrow (`input_ids`) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/offline/pt_text_dclm_1024_12k` | 12K seq, len=1024 | N/A | OK |
| 1a text-only (大) | 文本 PT | HF Arrow | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/offline/pt_text_fineweb_1024_100k` | 100K seq, len=1024 | N/A | OK |
| 1a text-only (overfit) | 文本 PT | HF Arrow | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/offline/pt_text_dclm_1024_overfit256` | 256 seq | N/A | OK |
| **1b image-only** | WDS latents | tar (npy+txt+json) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32` | ~subset, 128×128 | 16 | OK (单节点) |
| 1b image-only (全量) | WDS latents | tar (npy+txt+json) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_4096` | 67K, 128×128 | 17 | OK (单节点) |
| **2a mixed** | WDS latents | tar (npy+txt+json) | 同上 | 同上 | 同上 | OK (单节点) |
| **2b interleaved** | WDS latents | tar (npy+txt+json) | 同上 | 同上 | 同上 | OK (单节点) |
| **多节点版本** | WDS latents (resharded) | tar (npy+txt+json) | 待生成 | 同上 | >= 64 | **需 reshard** |
| **5 scale-up** | WDS latents (大规模) | tar (npy+txt+json) | 未准备 | CC12M / LAION 级 | >> 64 | **未准备** |

### 1.2 验证/评估数据状态

| 阶段 | 资源类型 | 路径 | 状态 |
|---|---|---|---|
| 1a text-only | 评估提示词 (minimal) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_text_minimal.jsonl` | OK (6 条) |
| 1a text-only | 评估提示词 (tiered) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_text_tiered_v1.jsonl` | OK (16 条) |
| 1a text-only | 批量评估脚本 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_prompts.py` | OK |
| 1a text-only | 离线 loss 评估 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_loss.py` | OK |
| 1a text-only | checkpoint sweep | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_checkpoint_sweep.py` | OK |
| 1b image-only | 图像评估提示词 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_image_v1.jsonl` | OK (16 条) |
| 1b image-only | 图像采样评估脚本 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_image_only/eval_image_sample.py` | OK |
| 1b image-only | 过拟合评估 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/overfit_eval_wds.py` | OK |
| 2a mixed | 混合评估提示词 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_mixed_v1.jsonl` | OK (16 条) |
| 2b interleaved | 三模式采样脚本 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py` | OK |
| 通用 | latent 解码验证 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/verify_wds_latents_decode.py` | OK |
| 通用 | 单条采样 + VAE decode | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/sample_and_decode.py` | OK |

### 1.3 Tokenizer 状态

| 路径 | 基底模型 | 特殊 Token | 状态 |
|---|---|---|---|
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer` | GPT-2 | `<\|oneflow_image\|>`, `<\|oneflow_image_som\|>`, `<\|oneflow_image_eom\|>` | OK |

---

## 2. 各阶段数据需求详解

### 2.1 Phase 1a: Text-only（已就绪）

**训练数据格式**: HuggingFace Datasets (`save_to_disk`)
- 字段: `input_ids`（List[int]，固定长度 1024）
- 加载方式: `--dataset_args <path>/dataset --load_preprocessed_data True`

**评估数据**: `eval_prompts_text_minimal.jsonl` / `eval_prompts_text_tiered_v1.jsonl`

**准备命令** (如需扩充数据):
```bash
python -u scripts/oneflow/prepare_pt_text_dataset.py \
  --dataset_name_or_path "mlfoundations/dclm-baseline-1.0" \
  --text_field text \
  --tokenizer_name_or_path data/latents_128_bundle/tokenizer \
  --seq_length 1024 \
  --streaming --train_limit 1000000 \
  --output_dir data/offline/pt_text_dclm_1024_1M
```

### 2.2 Phase 1b: Image-only

**训练数据格式**: WebDataset tar shards
- 每个 sample 包含: `.npy` (latent [4,16,16]), `.txt` (caption), `.json` (可选, `{"input_ids": [...]}`)
- Latent shape: `[4, H/8, W/8]`，对于 128×128 图像 → `[4, 16, 16]`
- Latent scale: `0.18215`

**当前数据**:
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32` (16 shards, 花朵子集)
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_4096` (17 shards, 67K CC3M 样本)

**多节点需求**: 4 nodes × 16 NPU = 64 processes → 需 >= 64 shards

**评估需求**: 图像采样 + VAE decode 可视化

### 2.3 Phase 2a: Mixed Generation

**训练数据格式**: 同 Phase 1b（WDS latent shards）
- 当前可将 `--mixed_generation_prob` 理解为 **mixed-generation 的入门级实验语义**：它意在表达“多少期望样本走含图像分支”。
- 但基础运行时尚不能过度表述为“已完整实现并验证：按 `mixed_generation_prob` 对图像侧做核心 mask/训练控制”；这部分仍待实现细化与专项验证。

**评估需求**: 同时评估文本和图像生成质量

### 2.4 Phase 2b: Interleaved

**训练数据格式**: 同 Phase 1b（WDS latent shards）
- 完整 Algorithm 3 调度: τ_text ~ Unif[0, τ_text_max]

**评估需求**: 三种采样模式（text-only, image-conditioned, full interleaved）

### 2.5 Phase 5: Scale-up（未来）

**目标数据规模**: CC12M (12M pairs) 或 LAION-400M 子集

**准备步骤**:
1. 导出 URL 列表
2. img2dataset 下载
3. VAE latent 编码
4. Resharding

详见第 5 节。

---

## 3. 多节点训练数据 Resharding

### 3.1 问题

当前 shard 数 (16-17) 不足以支持多节点训练（64+ processes）。WebDataset 按 shard 分发到各 rank，shard 数 < world_size 会导致部分 rank 没有数据。

### 3.2 解决方案

使用 `reshard_wds.py` 重新切分 tar，不需要重新计算 VAE latents。

### 3.3 Resharding 命令

当前 worktree 中可直接使用现有脚本 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/reshard_wds.py`；`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/reshard_for_multinode.sh` 在本 worktree 中未找到，可视为计划中的便捷 wrapper，而不是已存在入口。

推荐直接执行：

```bash
# flower32 子集: ~subset samples → maxcount=512 → 足够多的 shards
python -u scripts/oneflow/reshard_wds.py \
  --input_shards data/latents_128_bundle/wds_latents_flower32 \
  --output_dir data/latents_128_bundle/wds_latents_flower32_mn \
  --maxcount 512

# CC3M 全量 latents: 67K → maxcount=512 → ~131 shards
python -u scripts/oneflow/reshard_wds.py \
  --input_shards data/latents_128_bundle/wds_latents_4096 \
  --output_dir data/latents_128_bundle/wds_latents_4096_mn \
  --maxcount 512
```

### 3.4 验证

```bash
# 检查 shard 数量 >= 目标 world_size
ls data/latents_128_bundle/wds_latents_flower32_mn/*.tar | wc -l
ls data/latents_128_bundle/wds_latents_4096_mn/*.tar | wc -l
```

### 3.5 使用 resharded 数据训练

本 worktree 未提供现成的 image-only N-node launch shell；实践上可直接复用你现有的分布式启动方式，并把训练入口保持为 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_image_only/pt_image.py`，同时把 `--shards` 改为 resharded 目录，例如：
```bash
accelerate launch ... examples/oneflow_image_only/pt_image.py \
  --shards data/latents_128_bundle/wds_latents_flower32_mn \
  ...
```

---

## 4. 评估数据与脚本

### 4.1 文本评估（已就绪）

| 资源 | 路径 | 说明 |
|---|---|---|
| 提示词 (minimal) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_text_minimal.jsonl` | 6 条，含 expected_contains |
| 提示词 (tiered) | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_text_tiered_v1.jsonl` | 16 条，分层难度 |
| 批量评估 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_prompts.py` | JSONL → report.md |
| Loss 评估 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_loss.py` | 多 seed 离线 loss |
| Sweep | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_checkpoint_sweep.py` | 多 checkpoint 排名 |

**使用示例**:
```bash
python -u scripts/oneflow/eval_text_only_prompts.py \
  --model_dir data/ckpts/<experiment>/checkpoint-<step> \
  --prompts_file scripts/oneflow/eval_prompts_text_tiered_v1.jsonl \
  --output_dir outputs/eval_text/<experiment>
```

### 4.2 图像评估

| 资源 | 路径 | 说明 |
|---|---|---|
| 提示词 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_image_v1.jsonl` | 16 条，涵盖花朵 / 自然 / 物体 / 抽象 |
| 批量采样 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_image_only/eval_image_sample.py` | 批量采样 + VAE decode + report |
| 过拟合评估 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/overfit_eval_wds.py` | GT vs Gen 对比 + MSE/PSNR |
| 单条采样 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/sample_and_decode.py` | 单条提示 → 图像 |
| latent 验证 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/verify_wds_latents_decode.py` | 验证 latent shard decode 质量 |

**批量采样使用示例**:
```bash
python -u scripts/oneflow_image_only/eval_image_sample.py \
  --model_dir data/ckpts/image_only_overfit_1b1/checkpoint-5000 \
  --prompts_file scripts/oneflow/eval_prompts_image_v1.jsonl \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir outputs/eval_image/image_only_1b1
```

**过拟合评估使用示例**:
```bash
python -u examples/oneflow/overfit_eval_wds.py \
  --model_dir data/ckpts/image_only_overfit_1b1/checkpoint-5000 \
  --wds_shards data/latents_128_bundle/wds_latents_flower32 \
  --sample_index 0 \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir data/vis/overfit_eval_1b1
```

### 4.3 混合模态评估

| 资源 | 路径 | 说明 |
|---|---|---|
| 提示词 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_prompts_mixed_v1.jsonl` | 16 条，含纯文本和图像生成提示 |

使用现有的 `eval_text_only_prompts.py`（文本部分）和 `eval_image_sample.py`（图像部分）分别评估。

### 4.4 Interleaved 评估

| 资源 | 路径 | 说明 |
|---|---|---|
| 三模式采样 | `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py` | text_only / image_conditioned / interleaved |

**使用示例**:
```bash
# Text-only 模式
python -u examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
  --mode text_only \
  --prompt "OneFlow is a generative model" \
  --output_dir outputs/interleaved_text

# Image-conditioned 模式
python -u examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
  --mode image_conditioned \
  --prompt "a photo of a flower" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir outputs/interleaved_image

# Full interleaved 模式
python -u examples/oneflow_interleaved/sample_interleaved.py \
  --model_dir data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
  --mode interleaved \
  --prompt "a photo of a flower" \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir outputs/interleaved_full
```

---

## 5. 大规模数据准备路线（Scale-up）

### 5.1 完整流水线

```
HuggingFace 元数据 → urls.tsv → img2dataset → raw WDS → VAE encode → latent WDS → reshard
```

### 5.2 Step 1: 导出 URL

```bash
python -u scripts/oneflow/hf_export_urls_tsv.py \
  --dataset pixparse/cc12m-wds --split train \
  --output_tsv data/urls_cc12m.tsv \
  --max_rows 0  # 0=全部
```

### 5.3 Step 2: 下载图像

```bash
NO_ALBUMENTATIONS_UPDATE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 img2dataset \
  --url_list data/urls_cc12m.tsv \
  --input_format tsv --url_col url --caption_col caption \
  --output_format webdataset \
  --output_folder data/wds_128_cc12m \
  --image_size 128 \
  --processes_count 8 --thread_count 16 --timeout 15 --retries 2
```

### 5.4 Step 3: VAE Latent 编码

```bash
python -u scripts/oneflow/precompute_latents_wds.py \
  --input_shards "data/wds_128_cc12m/*.tar" \
  --output_dir data/latents_128_cc12m_bundle \
  --image_size 128 \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --batch_size 64 --num_workers 8 --maxcount 512 \
  --tokenizer_name_or_path data/latents_128_bundle/tokenizer \
  --max_caption_tokens 128 --write_input_ids True
```

### 5.5 Step 4: 验证

```bash
python -u scripts/oneflow/verify_wds_latents_decode.py \
  --shards data/latents_128_cc12m_bundle/wds_latents \
  --vae_id_or_path stabilityai/sd-vae-ft-mse \
  --output_dir data/vis/cc12m_decode_check \
  --max_samples 16
```

### 5.6 Step 5: Resharding（如需要）

```bash
python -u scripts/oneflow/reshard_wds.py \
  --input_shards data/latents_128_cc12m_bundle/wds_latents \
  --output_dir data/latents_128_cc12m_bundle/wds_latents_mn \
  --maxcount 512
```

---

## 6. 数据验证 Checklist

在开始每个阶段的实验前，请确认：

### Phase 1a (Text-only)
- [ ] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/offline/pt_text_*` 目录存在且非空
- [ ] `dataset/train/dataset.arrow` 文件可读
- [ ] tokenizer 与数据 tokenize 时使用的一致
- [ ] `eval_prompts_text_*.jsonl` 格式正确

### Phase 1b (Image-only)
- [ ] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/wds_latents_flower32/*.tar` 存在
- [ ] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/latents_128_bundle/tokenizer/` 可读
- [ ] `stats.json` 中 `latent_shape=[4,16,16]` 且 `latent_scale=0.18215`
- [ ] 分片数 >= world_size（16 个 NPU → 16 shards OK；多节点需 reshard）
- [ ] VAE decoder 可用（本地路径或 HF 缓存）
- [ ] `eval_prompts_image_v1.jsonl` 格式正确

### Phase 2a (Mixed)
- [ ] 同 Phase 1b 数据检查
- [ ] `eval_prompts_mixed_v1.jsonl` 格式正确

### Phase 2b (Interleaved)
- [ ] 同 Phase 1b 数据检查
- [ ] `sample_interleaved.py` 可用

### 多节点训练
- [ ] resharded 目录 shard 数 >= num_machines × processes_per_machine
- [ ] 所有节点可访问同一数据路径（共享存储）

---

## 7. 工具脚本索引

### 数据准备
| 脚本 | 功能 |
|---|---|
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/hf_export_urls_tsv.py` | HF 元数据 → urls.tsv |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/precompute_latents_wds.py` | 图像 → VAE latent WDS shards |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/reshard_wds.py` | 重分片（不重算 latent） |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/reshard_for_multinode.sh` | 计划中的一键多节点 resharding wrapper（本 worktree 未找到） |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/prepare_pt_text_dataset.py` | 文本 tokenize → HF Arrow |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/prepare_pt_bundle.sh` | 文本 PT 数据准备 wrapper |

### 数据验证
| 脚本 | 功能 |
|---|---|
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/verify_wds_latents_decode.py` | 解码 latent 验证质量 |

### 评估
| 脚本 | 功能 |
|---|---|
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_prompts.py` | 文本提示批量评估 |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_loss.py` | 离线 loss 评估 |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow/eval_text_only_checkpoint_sweep.py` | 多 checkpoint 排名 |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_image_only/eval_image_sample.py` | 图像采样 + VAE decode 批量评估 |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/overfit_eval_wds.py` | GT vs Gen 过拟合评估 |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow/sample_and_decode.py` | 单条采样 + VAE decode |
| `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py` | Interleaved 三模式采样 |
