# OneFlow H200 离线执行手册：PT + 伪 SFT + 评测（中文）

本文档面向 H200（CUDA）离线集群，目标是先搭建可复现的 text-only 训练与评测闭环：

1. 数据路径就绪检查  
2. PT 离线训练  
3. 基于 PT 样本构建伪 SFT 数据  
4. 伪 SFT 训练  
5. loss + prompt 批量评测  

> 说明：伪 SFT 仅用于 pipeline 对齐和稳定性验证，不等价于真实指令数据效果。

---

## 0. 环境激活（H200）

项目已提供 H200 推荐环境脚本：

```bash
source init_env.sh
```

等价内容见 `init_env.sh`：
- `wandb offline`
- CUDA 路径导出
- conda 环境激活

---

## 1. 路径规范（强烈建议统一）

建议统一使用以下变量：

```bash
export FINEWEB_ROOT=/mnt/shared-storage-user/ai4sreason/zhangjinouwen/huggingface/fineweb-edu_sample-10BT
export PT_BUNDLE=/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/dllm/oneflow/dllm/data/offline/pt_text_fineweb_edu_100k
export SFT_BUNDLE=/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/dllm/oneflow/dllm/data/offline/sft_text_pseudo_from_pt_100k
export TOKENIZER_DIR=$PT_BUNDLE/tokenizer
export CKPT_DIR=/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/dllm/oneflow/dllm/data/ckpts
```

---

## 2. fineweb 本地可用性检查（只读）

新增脚本：`scripts/oneflow/check_fineweb_local_readiness.py`

```bash
python -u scripts/oneflow/check_fineweb_local_readiness.py \
  --fineweb_root "$FINEWEB_ROOT" \
  --output_json /tmp/fineweb_readiness.json
```

脚本会识别 3 种状态：
- `save_to_disk`：可直接作为 `--dataset_args`（配合 `--load_preprocessed_data True`）
- `parquet`：可用于转换成离线 PT bundle
- `metadata_only`：仅有 `.parquet.metadata`，不可直接训练

如果结果是 `metadata_only`，需要先拿到真实 parquet 或先构建 `save_to_disk` 数据目录。

---

## 3. 构建 PT 离线 bundle

推荐入口：
- `scripts/oneflow/prepare_pt_text_dataset.py`
- 包装脚本：`scripts/oneflow/prepare_pt_bundle.sh`

示例（优先使用本地 fineweb 路径）：

```bash
bash scripts/oneflow/prepare_pt_bundle.sh \
  --dataset_name_or_path "$FINEWEB_ROOT/sample/10BT" \
  --text_field text \
  --tokenizer_name_or_path gpt2 \
  --seq_length 1024 \
  --streaming False \
  --train_split train \
  --test_split None \
  --train_limit 100000 \
  --output_dir "$PT_BUNDLE"
```

如本地路径不可直接读取，再考虑在可联网环境用 HF ID 准备 bundle 后拷贝到离线集群。

离线环境下 tokenizer 建议使用本地路径（避免 `gpt2` 触发在线 HEAD 请求）：

```bash
export GPT2_LOCAL=/root/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

并在准备命令中替换：

```bash
--tokenizer_name_or_path "$GPT2_LOCAL"
```

---

## 4. H200 上启动 PT 训练（离线 + wandb offline + tensorboard）

新增脚本：`scripts/oneflow/launch_pt_text_h200.sh`

```bash
bash scripts/oneflow/launch_pt_text_h200.sh \
  --pt_bundle "$PT_BUNDLE" \
  --output_dir "$CKPT_DIR/oneflow_text_pt_h200_fineweb" \
  --num_processes 8 \
  --max_steps 2000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1
```

默认行为：
- `WANDB_MODE=offline`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `--report_to wandb tensorboard`
- tensorboard 日志目录：`<output_dir>/tensorboard`

查看 TensorBoard：

```bash
tensorboard --logdir "$CKPT_DIR/oneflow_text_pt_h200_fineweb/tensorboard"
```

### 4.1 推荐：用一键脚本切换模型规模（0.6B~1.3B）

如果你希望直接用项目入口脚本（带自动数据准备、`TARGET_EPOCHS`、4/8 卡自适应），可使用：

```bash
PT_BUNDLE="$PT_BUNDLE" \
PT_TRAIN_LIMIT=100000 \
TARGET_EPOCHS=1 \
MAX_LENGTH=1024 \
MODEL_SIZE_PRESET=0p9b \
./run_pt_fineweb_edu.sh
```

可选规模：
- `MODEL_SIZE_PRESET=0p6b`（约 0.57B）
- `MODEL_SIZE_PRESET=0p9b`（约 0.88B）
- `MODEL_SIZE_PRESET=1p1b`（约 1.10B）
- `MODEL_SIZE_PRESET=1p3b`（约 1.35B）
- `MODEL_SIZE_PRESET=custom` + 手动 `MODEL_DIM/MODEL_DEPTH/MODEL_HEADS`

---

## 5. 从 PT 样本构建伪 SFT bundle

新增脚本：`scripts/oneflow/build_pseudo_sft_from_pt.py`

```bash
python -u scripts/oneflow/build_pseudo_sft_from_pt.py \
  --pt_bundle "$PT_BUNDLE" \
  --output_dir "$SFT_BUNDLE" \
  --strategy deterministic_uniform \
  --prompt_min_tokens 32 \
  --prompt_max_tokens 512 \
  --max_length 1024
```

输出目录结构：
- `$SFT_BUNDLE/dataset`
- `$SFT_BUNDLE/tokenizer`

---

## 6. H200 上启动伪 SFT 训练

新增脚本：`scripts/oneflow/launch_sft_text_h200.sh`

```bash
bash scripts/oneflow/launch_sft_text_h200.sh \
  --init_model_dir "$CKPT_DIR/oneflow_text_pt_h200_fineweb/checkpoint-final" \
  --sft_bundle "$SFT_BUNDLE" \
  --output_dir "$CKPT_DIR/oneflow_text_sft_h200_pseudo" \
  --num_processes 8 \
  --max_steps 1000 \
  --per_device_train_batch_size 4
```

如需与 PT 一键脚本联动，直接执行：

```bash
INIT_MODEL_DIR="$CKPT_DIR/oneflow_text_pt_h200_fineweb/checkpoint-final" \
MODEL_SIZE_PRESET=auto \
./run_sft_fineweb_edu.sh
```

说明：
- `MODEL_SIZE_PRESET=auto` 会优先读取 `INIT_MODEL_DIR/oneflow_config.json` 自动继承模型结构。
- 若要强制覆盖，可设 `MODEL_SIZE_PRESET=0p9b/1p1b/1p3b` 或 `custom`。

---

## 7. 评测闭环（不依赖长训练）

### 7.1 loss 定量评测

复用脚本：`scripts/oneflow/eval_text_only_loss.py`

建议在 checkpoint 选点时使用稳定口径（多 seed + 足够 batch）：
- `num_batches >= 200`
- `seeds=41,42,43`（或 3 个固定 seed）

```bash
python -u scripts/oneflow/eval_text_only_loss.py \
  --model_dir "$CKPT_DIR/oneflow_text_pt_h200_fineweb/checkpoint-final" \
  --dataset_dir "$PT_BUNDLE/dataset" \
  --device cuda \
  --batch_size 8 \
  --num_batches 200 \
  --seeds 41,42,43 \
  --compare_random True \
  --output_json "$CKPT_DIR/oneflow_text_pt_h200_fineweb/eval/loss_eval_multiseed.json"
```

### 7.2 批量 prompt 评测

新增脚本：`scripts/oneflow/eval_text_only_prompts.py`  
模板集：`scripts/oneflow/eval_prompts_text_minimal.jsonl`

```bash
python -u scripts/oneflow/eval_text_only_prompts.py \
  --model_dir "$CKPT_DIR/oneflow_text_pt_h200_fineweb/checkpoint-final" \
  --prompts_file scripts/oneflow/eval_prompts_text_minimal.jsonl \
  --output_dir "$CKPT_DIR/oneflow_text_pt_h200_fineweb/prompt_eval" \
  --device cuda
```

输出：
- `report.jsonl`
- `report.md`

### 7.3 PT 多 checkpoint 稳定评测与自动选点

新增脚本：`scripts/oneflow/eval_text_only_checkpoint_sweep.py`

用途：
- 对 `checkpoint-*` 批量执行 multi-seed loss 评测
- 可选批量 prompt 评测
- 输出 `summary.json` + `summary.md`，并给出 best checkpoint

```bash
python -u scripts/oneflow/eval_text_only_checkpoint_sweep.py \
  --ckpt_root "$CKPT_DIR/oneflow_text_pt_h200_fineweb" \
  --dataset_dir "$PT_BUNDLE/dataset" \
  --prompts_file scripts/oneflow/eval_prompts_text_minimal.jsonl \
  --output_dir "$CKPT_DIR/oneflow_text_pt_h200_fineweb/stable_eval" \
  --device cuda \
  --loss_batch_size 8 \
  --loss_num_batches 200 \
  --loss_seeds 41,42,43 \
  --run_prompt_eval True \
  --prompt_device cuda \
  --ranking_mode hybrid
```

可选排序模式：
- `hybrid`：优先 prompt pass rate（若可用），再看 `loss_tok`
- `loss_tok`：仅按 `loss_text_tok` 均值排序
- `loss_total`：仅按 `loss_total` 均值排序
- `prompt_first`：先 prompt，再 `loss_tok`

---

## 8. 常见问题速查

1) `fineweb` 路径只有 metadata，训练脚本报找不到数据  
- 先跑 `check_fineweb_local_readiness.py` 确认状态  
- `metadata_only` 不能直接训练，需切换到 parquet 或 `save_to_disk` 目录

2) 离线环境仍尝试访问外网  
- 确认 `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1`

3) wandb 上报失败  
- 确认 `WANDB_MODE=offline`  
- 若完全不需要 wandb，可改为 `--report_to none`

4) TensorBoard 无日志  
- 确认 `--report_to wandb tensorboard`  
- 检查 `--logging_dir <output_dir>/tensorboard` 是否存在事件文件
