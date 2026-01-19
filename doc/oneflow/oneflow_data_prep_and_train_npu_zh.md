# OneFlow 多模态：数据准备 → 离线 latents → 16×Ascend NPU 本地训练（中文）

本文档整理了从 **HuggingFace 元数据 → `img2dataset` 下载 → VAE 预计算 latents → 本地 16×Ascend NPU 训练** 的一整套流程，并包含本次 bring-up 中遇到的关键坑位与对应修复。

如需排查“32 张小数据集仍无法过拟合/训练-采样不匹配/数据集 decode 是否正确”等问题，请看：
- `doc/oneflow/oneflow_overfit_debug_zh.md`

> 适用场景：
> - 本地无 Slurm（单机多卡）
> - 训练在离线/内网环境（只读本地 `wds_latents/` + `tokenizer/`）
> - Ascend NPU（torch_npu）

---

## 0. 目录与脚本位置（本仓库已提供）

### 数据准备
- HuggingFace 导出 `urls.tsv`：`scripts/oneflow/hf_export_urls_tsv.py`
- 下载图片（WebDataset shards）：`img2dataset`（外部包）
- 预计算 VAE latents：`scripts/oneflow/precompute_latents_wds.py`
- 重新分片（把 tar 切得更细，不重算）：`scripts/oneflow/reshard_wds.py`

### 训练入口
- 训练：`examples/oneflow/pt_wds_latents.py`
- 本地 NPU accelerate 配置：`scripts/accelerate_configs/npu_ddp.yaml`

---

## 1. 环境准备（Ascend / 代理 / 离线）

### 1.1 Ascend 环境（按机器实际路径调整）

```bash
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_NPU_SOCKET_PORT_RANGE=auto

# 推荐
export PYTHONPATH=.:$PYTHONPATH
```

### 1.2 离线训练（建议）

训练时如果完全使用本地 tokenizer + 本地 shards，可设置：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 1.3 代理（仅下载/HF 导出时需要）

如果需要代理访问外网，按你们环境设置 `http_proxy/https_proxy`（略）。

---

## 2. 从 HuggingFace 导出 `urls.tsv`

### 2.1 关键点（避免踩坑）

- **不要用 `conceptual_captions --config 3m`**：你当前 `datasets` 版本里可用 config 是 `unlabeled/labeled`。
  - `unlabeled` 对应 CC3M（常用）
- 脚本默认 **streaming**（不全量下载 dataset），更省磁盘。
- 某些环境在 Python 退出阶段会出现 `PyGILState_Release` 崩溃；脚本已改为写完后硬退出，文件仍然是完整的。

### 2.2 示例命令（CC3M）

在仓库根目录运行：

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm
mkdir -p data

python -u scripts/oneflow/hf_export_urls_tsv.py \
  --dataset conceptual_captions --config unlabeled --split train \
  --output_tsv data/urls.tsv \
  --max_rows 100000
```

验收：

```bash
ls -lh data/urls.tsv
wc -l data/urls.tsv
head -n 3 data/urls.tsv
```

`urls.tsv` 必须是 TSV（tab 分隔），并包含表头 `url<TAB>caption`。

---

## 3. 使用 `img2dataset` 下载图片（输出 WebDataset）

### 3.1 关键点（避免踩坑）

- **不要用 `python -m img2dataset`**：`img2dataset==1.47.0` 没有 `__main__.py`。
  - 正确方式：直接用 CLI：`img2dataset ...`
- 如果遇到 Albumentations 更新检查网络超时告警，建议关闭：
  - `NO_ALBUMENTATIONS_UPDATE=1`
- 如果在非 NPU 环境导入 `torch` 会触发 `torch_npu/libhccl.so` 相关错误，可临时关闭：
  - `TORCH_DEVICE_BACKEND_AUTOLOAD=0`

### 3.2 推荐下载命令（成功率与速度更均衡）

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm
mkdir -p data/wds_128_retry

NO_ALBUMENTATIONS_UPDATE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 img2dataset \
  --url_list /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/urls.tsv \
  --input_format tsv \
  --url_col url \
  --caption_col caption \
  --output_format webdataset \
  --output_folder /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/data/wds_128_retry \
  --image_size 128 \
  --processes_count 8 \
  --thread_count 16 \
  --timeout 15 \
  --retries 2 \
  --user_agent_token "Mozilla/5.0 (X11; Linux) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
```

验收：

```bash
ls -lh data/wds_128_retry | head
cat data/wds_128_retry/00000_stats.json
```

> 提示：CC3M/老 URL 集通常失败率较高；并发过大（如 32×64）反而会把代理/出口打爆，失败率飙升。

---

## 4. 预计算 VAE latents（强烈推荐）

### 4.1 关键点（避免踩坑）

- `img2dataset` 输出的 tar 通常是 `00000.tar/00001.tar/...`（不是 `shard-*.tar`）。
- 训练用 16×NPU 时，建议 **latents shard 数 >= world_size**，更推荐 **>= world_size × dataloader_num_workers**：
  - 否则会出现 `rank` 或 `DataLoader worker` 拿不到 shard，导致训练报错或吞吐下降。
- 如果你还没跑 latents 预计算，建议直接把 `--maxcount` 设小一些（例如 1024/2048），一次性生成足够多的 shard。

### 4.2 推荐命令（直接生成足够多的 latents shards）

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm

python -u scripts/oneflow/precompute_latents_wds.py \
  --input_shards "data/wds_128_retry/*.tar" \
  --output_dir "data/latents_128_bundle" \
  --image_size 128 \
  --vae_id_or_path "stabilityai/sd-vae-ft-mse" \
  --batch_size 64 \
  --num_workers 8 \
  --maxcount 1024 \
  --tokenizer_name_or_path "gpt2" \
  --max_caption_tokens 128 \
  --write_input_ids True
```

产物：
- `data/latents_128_bundle/wds_latents/*.tar`
- `data/latents_128_bundle/tokenizer/`
- `data/latents_128_bundle/stats.json`

---

## 5. 如果 shard 还是不够：重新分片（不重算）

如果你已经生成了 latents，但 shard 太少（例如只有 7 个 tar），可用以下脚本把 tar 切得更细：

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm

python -u scripts/oneflow/reshard_wds.py \
  --input_shards "data/latents_128_bundle/wds_latents" \
  --output_dir "data/latents_128_bundle/wds_latents_1024" \
  --maxcount 1024
```

训练时把 `--shards` 改成新目录即可。

---

## 6. 本地 16×Ascend NPU 训练（accelerate）

### 6.1 关键点（避免踩坑）

- 必须使用 `accelerate` 的 `MULTI_NPU`：
  - 配置已提供：`scripts/accelerate_configs/npu_ddp.yaml`
- 本仓库已修复 WebDataset 分布式读取的两个关键坑：
  - `nodesplitter=None`（避免 `single_node_only` 在 world_size>1 时直接报错）
  - `empty_check=False`（当 `num_workers` > shard 数导致某些 worker 空时，不再报错）
- 但仍建议 **先把 `--dataloader_num_workers 0` smoke test**，跑通后再逐步调大。
- 如不希望 wandb 上报：务必传 `--report_to none`（或设置 `WANDB_MODE=offline`）。

### 6.2 启动命令（smoke test）

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm

export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

accelerate launch \
  --config_file scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  examples/oneflow/pt_wds_latents.py \
  --output_dir "data/ckpts/stage3b_mm_latents_128" \
  --tokenizer_name_or_path "data/latents_128_bundle/tokenizer" \
  --shards "data/latents_128_bundle/wds_latents_1024" \
  --use_precomputed_ids True \
  --max_steps 50 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 0 \
  --learning_rate 1e-4 --warmup_ratio 0.01 \
  --dim 512 --depth 8 --heads 8 --dim_head 64 --dim_latent 4 \
  --report_to none \
  --save_strategy no \
  --logging_steps 1
```

### 6.3 正式训练（示例）

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/npu_ddp.yaml \
  --main_process_port 29500 \
  examples/oneflow/pt_wds_latents.py \
  --output_dir "data/ckpts/stage3b_mm_latents_128" \
  --tokenizer_name_or_path "data/latents_128_bundle/tokenizer" \
  --shards "data/latents_128_bundle/wds_latents_1024" \
  --use_precomputed_ids True \
  --max_steps 50000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 2 \
  --learning_rate 1e-4 --warmup_ratio 0.01 \
  --dim 512 --depth 8 --heads 8 --dim_head 64 --dim_latent 4 \
  --report_to none \
  --save_strategy steps --save_steps 2000 --save_total_limit 5 \
  --logging_steps 20
```

---

## 7. 常见报错速查

### 7.1 `No module named img2dataset.__main__`
- 原因：版本不支持 `python -m img2dataset`
- 解决：用 CLI：`img2dataset ...`

### 7.2 `BuilderConfig '3m' not found`
- 原因：`datasets` 版本下 conceptual_captions 的 config 名不叫 `3m`
- 解决：用 `--config unlabeled`（CC3M）

### 7.3 `Shard split produced empty shard list for rank=... world=...`
- 原因：`num_shards < world_size`
- 解决：增大 shard 数（`--maxcount` 设小）或用 `reshard_wds.py`

### 7.4 `No samples found in dataset; perhaps you have fewer shards than workers`
- 原因：每个 rank 的 shard 数 < `dataloader_num_workers`
- 解决：
  - 先 `--dataloader_num_workers 0` 跑通
  - 再增加 shard 数（`--maxcount` 更小 / re-shard）


