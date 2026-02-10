# OneFlow 训练数据集推荐（中文）

本文档整理了 OneFlow 各阶段训练推荐的公开数据集，以及数据规模与算力的匹配建议。

---

## 1. Text-only 预训练数据集

### 1.1 首选数据集（质量高、广泛使用）

| 数据集 | HuggingFace ID | 规模 | 语言 | 说明 |
|---|---|---|---|---|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 1.3T tokens | 英文 | 教育内容过滤，质量最高；适合小模型快速验证 |
| **DCLM-Baseline** | `mlfoundations/dclm-baseline-1.0` | 3T tokens | 英文 | DataComp-LM 优选子集，经过严格质量筛选 |
| **FineWeb** | `HuggingFaceFW/fineweb` | 15T tokens | 英文 | Common Crawl 全量清洗，规模最大 |

### 1.2 代码 / 多语言 / 混合数据集

| 数据集 | HuggingFace ID | 规模 | 语言 | 说明 |
|---|---|---|---|---|
| StarCoder Data | `bigcode/starcoderdata` | 250B tokens | 代码 | 86 种编程语言，适合代码能力训练 |
| The Stack v2 | `bigcode/the-stack-v2` | 900B tokens | 代码 | StarCoder2 训练数据 |
| CulturaX | `uonlp/CulturaX` | 6.3T tokens | 167 种语言 | mC4 + OSCAR 清洗合并，适合多语言 |
| SlimPajama | `cerebras/SlimPajama-627B` | 627B tokens | 英文 | RedPajama 去重精简版 |
| RedPajama v2 | `togethercomputer/RedPajama-Data-V2` | 30T tokens | 5 种语言 | Common Crawl + 质量信号标注 |

### 1.3 中文数据集

| 数据集 | HuggingFace ID | 规模 | 说明 |
|---|---|---|---|
| WuDaoCorpora | `p208p2002/wudao` | 200B tokens | 中文通用语料 |
| SkyPile | `Skywork/SkyPile-150B` | 150B tokens | 中文网页清洗 |
| ChineseWebText | `CASIA-LM/ChineseWebText` | 1.4T tokens | 中文网页，含质量评分 |

---

## 2. 多模态预训练数据集（图文对）

| 数据集 | HuggingFace ID | 规模 | 说明 |
|---|---|---|---|
| **CC12M** | `pixparse/cc12m-wds` | 12M pairs | 论文使用，过滤后的 Conceptual Captions |
| LAION-400M | `laion/laion400m-data` | 400M pairs | 大规模图文对 |
| LAION-Aesthetics | `laion/laion2B-en-aesthetic` | ~600M pairs | 美学评分过滤子集 |
| DataComp-1B | `mlfoundations/datacomp_1b` | 1.4B pairs | DataComp 竞赛数据集 |
| YFCC15M | - | 15M pairs | 论文使用（需从 YFCC100M 筛选） |

> 论文（arXiv:2510.03506）使用的预训练数据：filtered CC12M + YFCC + licensed data，共约 400M image-text pairs。

---

## 3. SFT 数据集（指令微调）

### 3.1 Text-only SFT

| 数据集 | HuggingFace ID | 规模 | 说明 |
|---|---|---|---|
| UltraChat 200K | `HuggingFaceH4/ultrachat_200k` | 200K 对话 | 高质量多轮对话 |
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | 1M 条 | 混合指令数据 |
| SlimOrca | `Open-Orca/SlimOrca` | 518K 条 | GPT-4 蒸馏指令 |

### 3.2 多模态 SFT

| 数据集 | HuggingFace ID | 规模 | 说明 |
|---|---|---|---|
| LLaVA-Instruct-150K | `liuhaotian/LLaVA-Instruct-150K` | 150K | 图文指令微调 |
| ShareGPT4V | `Lin-Chen/ShareGPT4V` | 100K | 高质量图文对话 |

---

## 4. 数据规模与算力匹配建议

### 4.1 Text-only PT（seq_len=1024）

| 场景 | 样本数 | Token 总量 | 16 NPU (BS=32) | 64 NPU (BS=32) |
|---|---|---|---|---|
| Overfit 验证 | 12.8K | ~13M | 2000 ep ≈ 6h | 2000 ep ≈ 2h |
| 中等验证 | 100K | ~100M | 1 ep ≈ 3min | 1 ep ≈ 1min |
| 小规模 PT | 1M | ~1B | 1 ep ≈ 30min | 1 ep ≈ 8min |
| 中规模 PT | 10M | ~10B | 1 ep ≈ 5h | 1 ep ≈ 1.3h |

### 4.2 每 epoch batch 数建议

为保证 SGD 梯度质量，建议每 device 每 epoch 至少有 **25-100 个 batch**：

```
N_min = num_devices × batch_size × target_batches_per_epoch

示例（16 NPU, BS=32, 100 batches/epoch）：
  N_min = 16 × 32 × 100 = 51,200
```

| 设备规模 | BS | 100 batch/epoch 所需 N | 25 batch/epoch 所需 N |
|---|---|---|---|
| 16 NPU | 8 | 12,800 | 3,200 |
| 16 NPU | 32 | 51,200 | 12,800 |
| 64 NPU | 8 | 51,200 | 12,800 |
| 64 NPU | 32 | 204,800 | 51,200 |

---

## 5. 数据准备命令

### 5.1 Text PT 数据（在线下载 + tokenize）

```bash
# FineWeb-Edu 100K 样本
bash scripts/oneflow/prepare_pt_bundle.sh \
  --dataset_name_or_path "HuggingFaceFW/fineweb-edu" \
  --text_field text \
  --tokenizer_name_or_path <TOKENIZER_PATH> \
  --seq_length 1024 \
  --streaming \
  --train_limit 100000 \
  --output_dir data/offline/pt_text_fineweb_edu_100k
```

```bash
# DCLM-Baseline 1M 样本
bash scripts/oneflow/prepare_pt_bundle.sh \
  --dataset_name_or_path "mlfoundations/dclm-baseline-1.0" \
  --text_field text \
  --tokenizer_name_or_path <TOKENIZER_PATH> \
  --seq_length 1024 \
  --streaming \
  --train_limit 1000000 \
  --output_dir data/offline/pt_text_dclm_1M
```

### 5.2 Text SFT 数据

```bash
bash scripts/oneflow/prepare_sft_bundle.sh \
  --dataset_args "HuggingFaceH4/ultrachat_200k" \
  --tokenizer_name_or_path <TOKENIZER_PATH> \
  --max_length 1024 \
  --mask_prompt_loss \
  --output_dir data/offline/sft_text_ultrachat
```

### 5.3 多模态数据（图文 WebDataset latents）

详见 `doc/oneflow/guides/oneflow_data_prep_and_train_npu_zh.md`。

---

## 6. 离线环境注意事项

- 所有数据集建议在有网络的环境中预先下载并 tokenize，生成 `dataset/` + `tokenizer/` 的 bundle 目录
- 训练时使用 `--load_preprocessed_data --dataset_args <bundle>/dataset` 加载
- 设置 `TRANSFORMERS_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 避免训练时意外联网
- tokenizer 与数据 tokenize 时使用的必须一致，否则 token id 不匹配
