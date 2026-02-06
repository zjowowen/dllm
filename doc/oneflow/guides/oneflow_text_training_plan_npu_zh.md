## OneFlow text-only：从“训练/推理链路正确”到“英文更正常 + 能答 Paris + 能写代码”的训练计划（Ascend NPU）

> 目标（短期）：
> - 先证明 **text-only 侧的训练确实学到了语言信号**（已用 `Eq(7) loss` 定量验证）
> - 再把模型能力推进到更实用：**更像正常英文**、能完成简单事实补全（如 Paris）、具备基本代码续写能力
> - 全程支持 **离线数据** 与 **多节点多卡** 训练

相关文档：
- 归零式验证（Stage 1 文本侧）：`doc/oneflow/validation/oneflow_zero_validation_zh.md`
- 论文规格（Eq. 7 文本 loss）：`doc/oneflow/design/oneflow_paper_spec_2510_03506.md`

---

## 0. 先澄清：为什么现在采样看起来“不像 GPT”

OneFlow 的 `OneFlowSampler` 是 **插入式（insertion）采样**，不是自回归 next-token 生成。
因此：
- 纯 prompt completion（如 “The capital of France is …”）在能力不够强/采样不合适时可能发散或重复高频词
- 这不代表训练没学到；更可靠的验证是 **训练目标本身的 loss**（仓库已提供 `eval_text_only_loss.py`）

后续想更接近“GPT 补全体验”，建议：
- 采样用 `--suppress_whitespace_tokens True` + `--max_seq_len` + `--max_new_tokens` 防退化
- 训练到一定规模后，再考虑更贴近 completion 的采样策略（append-only/末尾插入为主）

---

## 1. 训练路线（推荐两阶段：PT → SFT）

### 1.1 阶段 A：Text-only PT（预训练，学语言与代码“底座”）

**数据建议（混合）**
- **通用英文网页/新闻/论坛**：提升自然语言与事实覆盖（Paris 这类）
- **代码语料**：提升写代码能力（Python/JS 等）
- 经验上：
  - 只跑通用网页：写代码会很弱
  - 只跑代码：英文叙述会很怪

**可选数据源（示例）**
- Web：`mlfoundations/dclm-baseline-1.0` / FineWeb（按你们可获得源替换）
- Code：FineWeb-code / The Stack（按你们可获得源替换）

**离线准备（强烈推荐）**
1) 在“能联网的机器”把 raw dataset 预处理成离线 PT 格式（`save_to_disk`）：
   - 脚本：`scripts/oneflow/prepare_pt_text_dataset.py`
   - 输出：`<bundle>/dataset` + `<bundle>/tokenizer`

2) 把 `<bundle>` 整个目录拷贝到训练机/集群（避免训练时访问 Hub）

**离线 PT 数据准备示例（debug 子集）**
```bash
python -u scripts/oneflow/prepare_pt_text_dataset.py \
  --dataset_name_or_path mlfoundations/dclm-baseline-1.0 \
  --train_split train \
  --test_split None \
  --text_field text \
  --tokenizer_name_or_path gpt2 \
  --seq_length 1024 \
  --streaming True \
  --train_limit 20000 \
  --output_dir /tmp/pt_text_dclm_1024_dbg
```

**训练入口**
- `examples/oneflow/pt_text.py`
- 离线训练务必带：`--load_preprocessed_data True --streaming False`

---

### 1.2 阶段 B：Text-only SFT（指令/问答/代码风格对齐）

如果你的目标是“能答 Paris/写代码（按人类指令格式）”，单纯 PT 往往不够，建议加一段 SFT：
- SFT 数据里要包含：
  - QA/常识问答（让模型习惯用一句话回答）
  - 代码指令（“写一个 Python 函数…”）

训练入口可用：
- `examples/oneflow/sft_mm.py`（目前是 text-only baseline 的 SFT；后续可扩展多模态）

离线方式同样推荐：
- 在联网机器 `load_dataset(...)` 后 `save_to_disk(...)`
- 训练时 `--load_preprocessed_data True`

---

## 2. 训练量怎么定：用 token budget，而不是 epoch

text-only 训练推荐用 **token budget** 规划。

### 2.1 关键公式
- 每个 global step 的 token 数（近似）：
\[
\text{tokens/step} \approx \text{world\_size} \times \text{bs} \times \text{GA} \times \text{seq\_len}
\]
- 训练总 token：
\[
\text{tokens\_total} \approx \text{max\_steps} \times \text{tokens/step}
\]

### 2.2 经验量级（给你一个可落地的范围）
假设当前 toy 模型（`dim=512, depth=8`）是几十 M 参数量级：
- **快速可见提升**：1B–5B tokens
- **更“像正常英文”**：10B+ tokens
- **更稳定的代码续写**：需要 code 数据占比足够，且 token 数通常也要到 10B+ 级别

> 你现在的 debug 离线集只有 ~25M tokens（24820×1024），重复很多 epoch 也很难“变强”，因为数据覆盖太窄。

---

## 3. 验证训练/推理没问题（Checklist）

### 3.1 训练正确性（强推荐：量化）
- 定期跑：
  - `scripts/oneflow/eval_text_only_loss.py`
- 观察点：
  - `loss_total` 随训练降低
  - 同数据同 noising 下，trained 明显优于 random init

示例：
```bash
python -u scripts/oneflow/eval_text_only_loss.py \
  --model_dir /path/to/checkpoint-final \
  --dataset_dir /path/to/offline_bundle/dataset \
  --device npu \
  --batch_size 8 \
  --num_batches 50
```

### 3.2 推理链路（sanity prompts）
用 `examples/oneflow/sample.py`，建议加防退化选项：
- `--max_seq_len` / `--max_new_tokens`（防 OOM）
- `--suppress_whitespace_tokens True`（防空格塌缩）

示例：
```bash
python -u examples/oneflow/sample.py \
  --model_dir /path/to/checkpoint-final \
  --device npu \
  --prompt "The capital of France is" \
  --temperature 0.9 \
  --dt 0.05 \
  --max_steps 512 \
  --max_new_tokens 256 \
  --max_insertions_per_step 64 \
  --max_seq_len 512 \
  --max_w 20 \
  --image_num_tokens 0 \
  --suppress_whitespace_tokens True
```

> 预期：PT 后生成会更像自然语料；SFT 后才更可能稳定输出 “Paris.” 这类直接答案。

---

## 4. 多节点多卡训练（accelerate / HCCL）

本仓库提供了单机 16 卡配置：
- `scripts/accelerate_configs/npu_ddp.yaml`
以及 4 节点（4×16=64 卡）模板配置：
- `scripts/accelerate_configs/npu_ddp_4node.yaml`

多机时的关键是：
- 同一份代码与同一份离线数据在每台机器都可见（共享存储或复制到本地）
- 设置 `num_machines` / `machine_rank` / `main_process_ip` / `main_process_port`

建议你们集群先做 **2 节点 bring-up**：
- 目标：稳定跑 200–500 steps，不 hang、不 OOM、loss 正常下降
- 然后再扩到更多节点

### 4.1 4 节点（64 卡）启动模板（无 Slurm 场景）

在 **每个节点** 都执行同一条命令，只是 `--machine_rank` 不同：

```bash
cd /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm
source activate_python_env.sh
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

# 由你们环境提供/约定：
export MASTER_ADDR="<rank0 的 IP 或 hostname>"
export MASTER_PORT=29500
export NODE_RANK="<0..3>"

accelerate launch \
  --config_file scripts/accelerate_configs/npu_ddp_4node.yaml \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  examples/oneflow/pt_text.py \
  --output_dir "/path/to/ckpts/oneflow_text_pt_4n" \
  --tokenizer_name_or_path "/path/to/offline_bundle/tokenizer" \
  --dataset_args "/path/to/offline_bundle/dataset" \
  --load_preprocessed_data True \
  --streaming False \
  --max_length 1024 \
  --max_steps 200000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --warmup_ratio 0.01 \
  --eval_strategy no --do_eval False \
  --save_strategy steps --save_steps 2000 --save_total_limit 5 \
  --logging_steps 20 \
  --report_to none
```

> 建议：先把 `--max_steps` 改成 200 做 bring-up，确认 4 节点 HCCL 没问题再放大。

---

## 5. 推荐的“逐步放大”执行顺序（最稳）

1) **单机 16 卡**：离线 PT（小数据）确认训练/推理/评估全闭环
2) **单机 16 卡**：换更大离线 PT 数据（至少 10× 以上 token）
3) **多机 bring-up（2 节点）**：同样参数跑 500 steps
4) **多机规模化**：开始冲 token budget（10B+）
5) **SFT**：加 QA/代码指令数据，让模型“会答/会写”

如果你把目标规模（节点数、每节点卡数、计划训练天数）告诉我，我可以帮你把：
- `tokens_total`、`max_steps`、`global batch`、`LR`、`save_steps`、评估频率
这几项定成一套可直接运行的配置。

