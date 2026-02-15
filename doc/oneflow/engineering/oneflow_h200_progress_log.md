# OneFlow H200 实时进展日志（PT + 伪 SFT）

本文档用于记录 H200 离线集群上的 OneFlow text-only 研发进展。

记录建议：
- 每次变更记录“脚本/参数/数据/结果/结论/下一步”
- 保留失败实验（便于回溯）
- 每次实验命名唯一（`exp_id`）

---

## 路径约定

- `PT_BUNDLE`: PT 离线数据目录（含 `dataset/`、`tokenizer/`）
- `SFT_BUNDLE`: 伪 SFT 离线数据目录（含 `dataset/`、`tokenizer/`）
- `CKPT_DIR`: checkpoint 根目录

---

## 当前状态（初始化）

日期：2026-02-11

- 新增 `scripts/oneflow/check_fineweb_local_readiness.py`（fineweb 路径可用性检查）
- 新增 `scripts/oneflow/launch_pt_text_h200.sh`（H200 PT 启动脚本）
- 新增 `scripts/oneflow/build_pseudo_sft_from_pt.py`（PT -> 伪 SFT 转换）
- 新增 `scripts/oneflow/launch_sft_text_h200.sh`（H200 伪 SFT 启动脚本）
- 新增 `scripts/oneflow/eval_text_only_prompts.py` + `scripts/oneflow/eval_prompts_text_minimal.jsonl`（批量 prompt 评测）

---

## 实验记录模板

```text
## [exp_id] <实验标题>

### 背景
- 目标：
- 假设：

### 数据与模型
- PT_BUNDLE:
- SFT_BUNDLE:
- init_model_dir:

### 命令
- 粘贴完整命令（建议保留环境变量）

### 关键参数
- max_steps:
- bs / ga / global_bs:
- lr / warmup_ratio:
- sampler args:

### 结果
- 训练日志摘要：
- loss 曲线观察：
- prompt_eval 观察：

### 结论
- 是否达到预期：
- 问题定位：
- 下一步：
```

---

## 实验记录

## [h200_data_0001] fineweb 本地路径可用性检查（通过）

### 背景
- 目标：确认 `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/huggingface/fineweb-edu_sample-10BT` 是否可直接用于离线 PT 数据准备。
- 假设：路径中存在可读取的 parquet 文件，而非仅 metadata 缓存。

### 数据与模型
- PT_BUNDLE: 待创建
- SFT_BUNDLE: 待创建
- init_model_dir: N/A

### 命令
- `python -u scripts/oneflow/check_fineweb_local_readiness.py --fineweb_root "$FINEWEB_ROOT" --output_json /tmp/fineweb_readiness.json`

### 关键参数
- fineweb_root: `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/huggingface/fineweb-edu_sample-10BT`
- output_json: `/tmp/fineweb_readiness.json`

### 结果
- `ready=true`, `ready_mode=parquet`
- `parquet_files_count=14`
- parquet 示例路径位于：`sample/10BT/*.parquet`
- 同时检测到 `.cache/.../*.parquet.metadata`（非阻塞）

### 结论
- 数据路径可用于下一步 PT bundle 构建。
- 下一步：用 `prepare_pt_bundle.sh` 基于 `"$FINEWEB_ROOT/sample/10BT"` 生成 `PT_BUNDLE`，随后执行 H200 PT smoke 训练。

## [h200_smoke_0002] PT + 伪 SFT + 评测链路 smoke（完成）

### 背景
- 目标：在离线条件下验证从 fineweb parquet 到 PT/SFT/评测的全链路可执行性。
- 假设：本地 HF cache 可支持 tokenizer 离线加载，模型依赖齐全可完成最小训练步。

### 数据与模型
- PT_BUNDLE: `data/offline/pt_text_fineweb_edu_smoke4k`
- SFT_BUNDLE: `data/offline/sft_text_pseudo_from_pt_smoke4k`
- init_model_dir: `data/ckpts/oneflow_text_pt_cpu_smoke/checkpoint-final`

### 命令
- 首次 PT 数据准备（失败示例）：  
  `python -u scripts/oneflow/prepare_pt_text_dataset.py ... --tokenizer_name_or_path gpt2 ...`
- 修复后 PT 数据准备（成功）：  
  `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python -u scripts/oneflow/prepare_pt_text_dataset.py ... --tokenizer_name_or_path /root/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e ...`
- PT smoke 训练（CPU 回退，5 steps）：  
  `python -u examples/oneflow/pt_text.py ... --max_steps 5 --use_cpu True --bf16 False --optim adamw_torch --report_to none`
- 伪 SFT 构建：  
  `python -u scripts/oneflow/build_pseudo_sft_from_pt.py --pt_bundle ... --output_dir ... --strategy deterministic_uniform --prompt_min_tokens 32 --prompt_max_tokens 128 --max_length 256`
- SFT smoke 训练（CPU 回退，5 steps）：  
  `python -u examples/oneflow/sft_mm.py ... --max_steps 5 --use_cpu True --bf16 False --optim adamw_torch --report_to none`
- loss 评测：  
  `python -u scripts/oneflow/eval_text_only_loss.py --model_dir ... --dataset_dir ... --device cpu --batch_size 4 --num_batches 10 --compare_random True`
- prompt 批量评测（PT + SFT）：  
  `python -u scripts/oneflow/eval_text_only_prompts.py --model_dir ... --prompts_file scripts/oneflow/eval_prompts_text_minimal.jsonl --output_dir ... --device cpu`

### 关键参数
- PT 数据准备：`seq_length=256`, `train_limit=4000`, `num_proc=8`
- PT/SFT 训练：`max_steps=5`, `per_device_train_batch_size=2`, `dataloader_num_workers=0`
- 评测：`num_batches=10`, `prompt_count=5`

### 结果
- PT 数据准备首次失败：离线环境下 `gpt2` 触发 `huggingface.co` 超时重试。
- 修复后 PT 数据准备成功：产出 `dataset/` 与 `tokenizer/`，日志见 `data/offline/pt_text_fineweb_edu_smoke4k/prepare_pt_text_dataset.log`。
- PT smoke 训练成功：5 steps 完成，产出 `data/ckpts/oneflow_text_pt_cpu_smoke/checkpoint-final`。
- 伪 SFT 构建成功：产出 `data/offline/sft_text_pseudo_from_pt_smoke4k/dataset`。
- SFT smoke 训练成功：5 steps 完成，产出 `data/ckpts/oneflow_text_sft_cpu_smoke/checkpoint-final`。
- loss 评测输出（PT checkpoint）：
  - trained: `loss_total=8.0252`
  - random: `loss_total=5.6751`
  - 说明：仅 5-step smoke，不具备收敛意义。
- prompt 评测输出：
  - PT: `expected_contains_pass=0/4`（`prompt_eval/report.md`）
  - SFT: `expected_contains_pass=0/4`（`prompt_eval/report.md`）
  - 说明：当前只验证评测脚本可用与产物格式正确。

### 依赖修复记录
- 训练启动前补装：
  - `torchdiffeq`
  - `einx`, `ema-pytorch`, `axial-positional-embedding`, `rotary-embedding-torch`, `hyper-connections`, `loguru`, `jaxtyping`, `beartype`

### 结论
- 离线全链路脚本与产物路径均验证通过（数据准备、PT、伪 SFT、loss 评测、prompt 评测）。
- 关键注意点：离线环境不要用裸 `gpt2` repo id，需使用本地 tokenizer 路径或强制 offline 环境变量。
- 下一步：在 H200 可见 GPU 环境下复用同参数模板，将 `--use_cpu True` 切回默认 CUDA，并放大步数做真实收敛验证。

## [h200_gpu_0003] H200 2-GPU wrapper smoke（完成）

### 背景
- 目标：验证新增 H200 wrapper（PT/SFT）在真实 CUDA 多卡环境可直接运行。
- 假设：`launch_pt_text_h200.sh` 与 `launch_sft_text_h200.sh` 可在 2 张 H200 上完成短步训练并产出 checkpoint。

### 数据与模型
- PT_BUNDLE: `data/offline/pt_text_fineweb_edu_smoke4k`
- SFT_BUNDLE: `data/offline/sft_text_pseudo_from_pt_smoke4k`
- init_model_dir: `data/ckpts/oneflow_text_pt_h200_smoke/checkpoint-final`

### 命令
- PT wrapper（2 GPU, 10 steps）：
  - `bash scripts/oneflow/launch_pt_text_h200.sh --pt_bundle ... --output_dir data/ckpts/oneflow_text_pt_h200_smoke --num_processes 2 --max_steps 10 --max_length 256 --per_device_train_batch_size 2 --dataloader_num_workers 0`
- SFT wrapper（2 GPU, 10 steps）：
  - `bash scripts/oneflow/launch_sft_text_h200.sh --init_model_dir data/ckpts/oneflow_text_pt_h200_smoke/checkpoint-final --sft_bundle ... --output_dir data/ckpts/oneflow_text_sft_h200_smoke --num_processes 2 --max_steps 10 --max_length 256 --per_device_train_batch_size 2 --dataloader_num_workers 0`
- loss 评测（CUDA）：
  - `python -u scripts/oneflow/eval_text_only_loss.py --model_dir data/ckpts/oneflow_text_pt_h200_smoke/checkpoint-final --dataset_dir data/offline/pt_text_fineweb_edu_smoke4k/dataset --device cuda --batch_size 4 --num_batches 20 --compare_random True`
  - `python -u scripts/oneflow/eval_text_only_loss.py --model_dir data/ckpts/oneflow_text_sft_h200_smoke/checkpoint-final --dataset_dir data/offline/pt_text_fineweb_edu_smoke4k/dataset --device cuda --batch_size 4 --num_batches 20 --compare_random True`
- prompt 评测（CUDA）：
  - `python -u scripts/oneflow/eval_text_only_prompts.py --model_dir data/ckpts/oneflow_text_pt_h200_smoke/checkpoint-final --prompts_file scripts/oneflow/eval_prompts_text_minimal.jsonl --output_dir data/ckpts/oneflow_text_pt_h200_smoke/prompt_eval --device cuda`
  - `python -u scripts/oneflow/eval_text_only_prompts.py --model_dir data/ckpts/oneflow_text_sft_h200_smoke/checkpoint-final --prompts_file scripts/oneflow/eval_prompts_text_minimal.jsonl --output_dir data/ckpts/oneflow_text_sft_h200_smoke/prompt_eval --device cuda`

### 关键参数
- GPU: 2 x H200
- PT/SFT: `max_steps=10`, `bs=2/gpu`, `ga=1`, `logging_steps=1`, `save_steps=5`
- 日志：wandb offline + tensorboard（wrapper 默认）

### 结果
- 环境探测通过：`nvidia-smi` 可见 2 张 H200，`torch.cuda.is_available()=True`。
- PT wrapper 成功：`data/ckpts/oneflow_text_pt_h200_smoke/checkpoint-final` 生成。
- SFT wrapper 成功：`data/ckpts/oneflow_text_sft_h200_smoke/checkpoint-final` 生成。
- PT loss 评测（CUDA）：
  - trained: `loss_total=19.0491`
  - random: `loss_total=55.2156`
  - 说明：trained 明显优于 random（在当前采样下）。
- SFT loss 评测（CUDA, 对 PT 数据）：
  - trained: `loss_total=22.2478`
  - random: `loss_total=22.5720`
  - 说明：仅 10-step smoke，差异有限。
- prompt 评测（PT/SFT）：
  - PT: `expected_contains_pass=0/4`
  - SFT: `expected_contains_pass=0/4`
  - 说明：短步 smoke 仅验证评测流程，未到能力提升阶段。

### 结论
- H200 wrapper 与离线记录链路已在真实 2-GPU 环境贯通。
- 下一步应直接放大训练步数（例如 PT 2k+、SFT 1k+）并固定评测节奏观察能力变化。
- 训练脚本已增强为单机多卡自适应：`run_pt_fineweb_edu.sh`、`run_sft_fineweb_edu.sh` 在未显式指定 `NUM_PROCESSES` 时优先选择 8 卡，其次 4 卡（否则回退到可见卡数）。

## [h200_script_0004] 模型规模 preset 接线（0.6B~1.3B）

### 背景
- 目标：在不改训练 Python 主体代码的前提下，让 PT/SFT 一键脚本支持快速切换 0.6B~1.3B 规模。
- 假设：通过 shell wrapper 透传 `--dim/--depth/--heads/--dim_head/--dim_latent` 即可驱动模型规模变化。

### 数据与模型
- PT_BUNDLE: 复用现有路径
- SFT_BUNDLE: 复用现有路径
- init_model_dir: 复用现有 PT checkpoint；SFT 默认可从 `oneflow_config.json` 自动继承模型参数

### 命令
- 文件改造（无长训练）：
  - `run_pt_fineweb_edu.sh`
  - `run_sft_fineweb_edu.sh`
  - `scripts/oneflow/launch_pt_text_h200.sh`
  - `scripts/oneflow/launch_sft_text_h200.sh`
- 语法校验：
  - `bash -n run_pt_fineweb_edu.sh`
  - `bash -n run_sft_fineweb_edu.sh`
  - `bash -n scripts/oneflow/launch_pt_text_h200.sh`
  - `bash -n scripts/oneflow/launch_sft_text_h200.sh`

### 关键参数
- 新增环境变量：`MODEL_SIZE_PRESET`
  - `0p6b -> dim=896, depth=16, heads=14`
  - `0p9b -> dim=1024, depth=20, heads=16`
  - `1p1b -> dim=1152, depth=20, heads=18`
  - `1p3b -> dim=1280, depth=20, heads=20`
  - `custom -> 手动指定 MODEL_DIM/MODEL_DEPTH/MODEL_HEADS`
- 公共默认：`dim_head=64`、`dim_latent=4`

### 结果
- `launch_pt_text_h200.sh` / `launch_sft_text_h200.sh` 均已支持并透传模型结构参数。
- `run_pt_fineweb_edu.sh` 已支持基于 preset 一键切换模型规模，并打印参数与近似参数量提示。
- `run_sft_fineweb_edu.sh` 默认 `MODEL_SIZE_PRESET=auto`，会优先读取 `INIT_MODEL_DIR/oneflow_config.json` 自动继承模型结构，避免 PT/SFT 配置错配。
- 所有改造脚本通过 `bash -n` 语法检查。

### 结论
- 现在可用同一套入口脚本在 H200 上快速切换 0.6B~1.3B 规模进行 PT，并平滑衔接 SFT。
- 下一步：用 `MODEL_SIZE_PRESET=0p9b` 做 1 epoch 基线，再做 `1p1b/1p3b` 对比吞吐与 loss 曲线。

## [h200_eval_0005] 稳定评测与 checkpoint 自动选点工具（完成）

### 背景
- 目标：降低 PT 阶段 loss 抖动带来的误判风险，避免单次/小样本评测选错 checkpoint。
- 假设：使用多 seed + 大样本评测，并结合 prompt 结果排序，可显著提升选点稳定性。

### 数据与模型
- PT_BUNDLE: 复用现有 PT dataset（`$PT_BUNDLE/dataset`）
- checkpoints: 复用 `PT_OUT_DIR/checkpoint-*`
- prompts: `scripts/oneflow/eval_prompts_text_minimal.jsonl`

### 命令
- Python 语法检查：
  - `python -m py_compile scripts/oneflow/eval_text_only_loss.py scripts/oneflow/eval_text_only_checkpoint_sweep.py`
- Shell 语法检查：
  - `bash -n eval_sft_fineweb_edu.sh`

### 关键参数
- `eval_text_only_loss.py` 新增：
  - `--seeds "41,42,43"`（multi-seed）
  - `--output_json <path>`（结构化输出）
- `eval_sft_fineweb_edu.sh` 默认改为：
  - `EVAL_NUM_BATCHES=200`
  - `EVAL_SEEDS=41,42,43`
- 新增 sweep 脚本：
  - `scripts/oneflow/eval_text_only_checkpoint_sweep.py`
  - 支持 `ranking_mode=hybrid|loss_tok|loss_total|prompt_first`

### 结果
- `eval_text_only_loss.py` 保持单 seed 向后兼容，同时支持多 seed 聚合统计：
  - 每项输出 `mean/std/sem/ci95/min/p50/p95/max`
  - 可写入 `output_json`
- 新增 `eval_text_only_checkpoint_sweep.py`：
  - 自动遍历 `checkpoint-*`
  - 逐 checkpoint 跑 multi-seed loss eval（可选 compare random）
  - 可选 prompt eval
  - 输出 `summary.json` + `summary.md` + best checkpoint
- `eval_sft_fineweb_edu.sh` 已接入多 seed loss 评测，并可选触发 PT checkpoint sweep：
  - `RUN_PT_CKPT_SWEEP=1`

### 结论
- 现在可以在“训练完成后”快速得到稳定 checkpoint 排名，减少单次波动造成的决策误差。
- 下一步：在你当前 `1p3b@1024` 的 `checkpoint-*` 上跑一轮 sweep，固定 SFT 初始化点，再做对照 SFT。
