# OneFlow 910C 集群恢复进展

更新时间：2026-02-16

本文件记录在 910C 集群上承接 `doc/oneflow/PROGRESS.md` 的恢复执行结果，重点覆盖：
- 恢复计划执行状态（P0-P4）
- 当前最紧迫事项
- 历史堵塞点在 910C 的复现/解除状态

---

## 0) 本轮恢复结论（可直接执行）

- **已完成最小闭环并进入扩展阶段**：环境激活 -> 数据可读性确认 -> 50-step smoke -> 500-step baseline -> multi-seed 评测 -> checkpoint sweep -> 基于 best checkpoint 的 2k 续训 + prompt-sweep -> 2 节点 bring-up 启动器落地与链路验证。
- **910C 当前可用 batch 区间（text-only, 16 NPU, seq_len=1024）**：
  - `bs=32` 可跑（短程验证通过）
  - `bs=40/48` OOM（见下文）
- **CTMC 向量化在 910C 无性能退化**：与 paper loss 训练速度同量级。
- **离线风险确认**：`gpt2` repo id 在离线模式会失败；本地 tokenizer 路径可稳定运行。
- **最新选点状态（含单节点 1000-step 追加验证）**：
  - 历史 2k sweep 在原始 `hybrid`（prompt 优先）下曾出现 `checkpoint-400` 反转（高 prompt/高 loss）。
  - loss-guarded / `loss_tok` 主导口径下，主 baseline 一直为 `checkpoint-1600`（`loss_tok_mean=9.999`，`prompt_pass_rate=25.0%`）。
  - 在此基础上继续做单节点复核（从 `checkpoint-200` 再续 `500-step`，`lr=2e-5`）后，sweep 最优前移到 `checkpoint-100`（`loss_tok_mean=9.3210`, `loss_total_mean=8.0644`, `prompt_pass_rate=31.25%`）。
  - 相比 `checkpoint-1600`，`checkpoint-100` 增益更明确（`loss_tok` 约 `-6.78%`，`loss_total` 约 `-5.64%`，prompt `25.0% -> 31.25%`），当前建议更新为“`checkpoint-100` 候选主点 + `checkpoint-1600` 回退点（`checkpoint-200` 次级备选）”。

---

## 1) 集群环境与数据状态（P0）

### 1.1 环境检查
- 环境脚本来源：`init_env.sh`（Ascend + HCCL + 代理）
- 核验结果：
  - `ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest`
  - `HCCL_NPU_SOCKET_PORT_RANGE=auto`
  - `torch_npu` 可导入，`npu_available=True`，`npu_count=16`

### 1.2 数据可读性检查
- 数据根路径：
  - `/mnt/ai4s/zhangjinouwen/Dataset/hf_snapshots/fineweb-edu_sample-10BT/`
- 使用脚本：`scripts/oneflow/check_fineweb_local_readiness.py`
- 结果：
  - `ready=true`
  - `ready_mode=parquet`
  - `parquet_files_count=14`
  - 可直接进入 PT bundle/训练链路

### 1.3 tokenizer 策略（离线）
- 验证结论：
  - `AutoTokenizer.from_pretrained("gpt2")` 在离线模式失败（触发无法连接 huggingface）
  - `AutoTokenizer.from_pretrained("data/offline/pt_text_fineweb_1024_100k/tokenizer")` 成功
- 结论：910C 离线训练必须固定本地 tokenizer 路径，避免裸 repo id。

---

## 2) smoke 训练闭环（P1）

### 2.1 50-step PT smoke
- 输出目录：`data/ckpts/oneflow_text_pt_910c_smoke50`
- 关键参数：
  - `max_steps=50`
  - `per_device_train_batch_size=1`
  - `dataloader_num_workers=0`
  - `num_processes=16`（NPU）
- 结果：
  - 训练完成，`exit_code=0`
  - `train_runtime=23.9399s`
  - `train_steps_per_second=2.089`
  - `train_loss=12.0833`
  - 产物完整：`checkpoint-25`、`checkpoint-50`、`checkpoint-final`、tensorboard events

---

## 3) 历史堵塞点回归（P2）

### 3.1 OOM 门限扫描（16 NPU）
- 验证命令均为 text-only、`seq_len=1024`、`dataloader_num_workers=0`、短程 steps。
- 结果：
  - `bs=16`：通过
  - `bs=24`：通过
  - `bs=32`：通过
  - `bs=40`：失败，OOM（示例：`Tried to allocate 7.68 GiB`）
  - `bs=48`：失败，OOM（示例：`Tried to allocate 9.21 GiB`）
- 当前结论：
  - 910C 上稳定工作区间建议先用 `bs<=32`
  - 若要提高吞吐，优先通过 GA 提升 global batch，而非继续拉高 per-device BS

### 3.2 CTMC 向量化性能回归
- 对比设置：`bs=16`, `max_steps=20`, 仅切换 `text_loss_type`
- 结果：
  - paper：`train_steps_per_second=1.177`
  - ctmc：`train_steps_per_second=1.126`
- 结论：
  - CTMC 与 paper 同量级，无历史 `~62s/step` 级别退化
  - 向量化实现在 910C 验证通过

### 3.3 离线稳定性回归
- `WANDB_MODE=offline` 生效，训练/评测可持续运行
- 本地 tokenizer 路径离线可用，裸 `gpt2` repo id 离线不可用
- 结论：离线策略需作为默认模板固化

---

## 4) 短程基线 + 稳定选点（P3）

### 4.1 500-step baseline
- 输出目录：`data/ckpts/oneflow_text_pt_910c_baseline_s500`
- 关键参数：
  - `max_steps=500`
  - `per_device_train_batch_size=16`
  - `save_steps=100`
  - `save_total_limit=5`
- 结果：
  - 训练完成，`exit_code=0`
  - `train_runtime=342.0823s`
  - `train_steps_per_second=1.462`
  - `train_loss=20.4187`
  - 产物：`checkpoint-100/200/300/400/500` + `checkpoint-final`

### 4.2 multi-seed loss（checkpoint-final）
- 输出：
  - `data/ckpts/oneflow_text_pt_910c_baseline_s500/eval/loss_eval_multiseed.json`
- 设置：
  - `num_batches=100`
  - `seeds=41,42,43`
  - `compare_random=True`
- 关键结果（聚合均值）：
  - trained: `loss_total=19.4555`, `loss_tok=25.7672`
  - random: `loss_total=35.2514`, `loss_tok=33.7630`
- 结论：
  - trained 明显优于 random，训练目标有效学习成立

### 4.3 checkpoint sweep
- 输出：
  - `data/ckpts/oneflow_text_pt_910c_baseline_s500/stable_eval/summary.md`
  - `data/ckpts/oneflow_text_pt_910c_baseline_s500/stable_eval/summary.json`
- 设置：
  - `ranking_mode=loss_tok`
  - `loss_num_batches=30`
  - `loss_seeds=41,42,43`
  - `run_prompt_eval=False`
- 排名结果：
  1. `checkpoint-300` (`loss_tok_mean=15.5736`, `loss_total_mean=12.8241`)
  2. `checkpoint-400`
  3. `checkpoint-200`
  4. `checkpoint-500`
  5. `checkpoint-100`
- 当前最优初始化点：`checkpoint-300`

---

## 5) 当前最紧迫事项（按优先级）

1. **当前阶段先做单节点调试（按当前要求）**
   - 先在 1 节点上完成参数稳定性与选点口径收敛
   - 学习率短程调试优先 `2e-5`（`1e-4` 次之，`1e-5` 不建议）
   - 最新复核（见 11.10）显示：从 `checkpoint-200` 再续 500-step 后，`checkpoint-100` 进一步优于 `checkpoint-1600`
   - 多节点暂由外部手动启动（脚本已交付）
2. **补齐 2 节点实跑前置（集群侧）**
   - 需明确 rank1 节点地址/可达性（当前仅能解析 worker-0）
   - 两节点同时启动 `launch_pt_text_910c_2node.sh`，完成 200-500 steps bring-up
3. **以 `checkpoint-100` 为候选、`checkpoint-1600` 为回退继续验证**
   - 优先验证 `checkpoint-100` 在后续续训/多机场景下是否保持对 `checkpoint-1600` 的优势（`checkpoint-200` 作为次级备选）
   - tiered 集合下原始 hybrid 仍会把高 prompt/高 loss 的 checkpoint 提前
   - 当前生产建议：`loss_tok` 主导或加 loss-guard 后再用 prompt tie-break
4. **继续修正 prompt 口径**
   - 新 tiered 集已把 `pass_rate` 从 0 提升到可分辨区间
   - 但 hard 桶仍全 0，easy 桶占比过高，需继续打磨判别力
5. **固化评测模板**
   - 固定 seeds / num_batches / prompts_file / ranking 规则，避免选点漂移
6. **paper vs ctmc 长程对比**
   - 在多机链路稳定后补齐长程收敛和稳定性对比

---

## 6) 历史堵塞点状态判定（910C）

| 堵塞点 | H200 状态 | 910C 当前状态 | 备注 |
|---|---|---|---|
| OOM（高 BS） | 存在（BS=32 曾 OOM） | **部分复现** | 本轮 `bs32` 可过，`bs40/48` OOM；需按 910C 实测阈值执行 |
| CTMC 性能瓶颈 | 已通过向量化解除 | **已验证解除** | `ctmc` 与 `paper` 速度同量级 |
| 离线 tokenizer 触网风险 | 已识别 | **已复现并规避** | 裸 `gpt2` 离线失败，本地 tokenizer 路径可用 |
| 小数据收敛不稳 | 存在 | **风险仍在** | 本轮用 100k bundle，后续应避免回退到过小数据 |
| 评测选点波动 | 存在 | **风险仍在** | tiered prompt 下原始 hybrid 出现“高 prompt/高 loss”反转，需加 loss-guard |

---

## 7) 下一步 7 天执行清单

- Day 1-2：基于 `checkpoint-100` 做单节点 500/1000-step 续训复核，确认“短程再优化窗口”是否稳定复现（`checkpoint-1600` 作为回退）。
- Day 3：用固定口径（seeds/batches/prompts）再跑一次 guarded-hybrid sweep，验证选点稳定性。
- Day 4-5：2 节点 bring-up（200-500 steps，由外部手动启动）并记录通信稳定性与吞吐。
- Day 6-7：收敛主配置（`checkpoint-100`/`checkpoint-1600`、paper vs ctmc、batch/GA、save/eval cadence）并更新本文件。

---

## 8) 数据需求（当前）

- 目前 `/mnt/ai4s/zhangjinouwen/Dataset/hf_snapshots/fineweb-edu_sample-10BT/` 足够支撑下一轮 text-only 扩展。
- 如需推进“更强代码能力 + 指令跟随”，建议补充：
  - code 占比更高的离线 text-only bundle（100k+）
  - QA/指令风格 SFT 离线数据（用于后续 SFT 阶段）

---

## 9) Day 1-3 执行更新（已完成）

### 9.1 训练脚本补丁（支持 checkpoint 热启动）
- 文件：`examples/oneflow/pt_text.py`
- 变更：
  - `ModelArguments` 新增 `init_model_dir`
  - 当传入该参数时，使用 `OneFlowModel.from_pretrained(init_model_dir)` 初始化模型
- 目的：
  - 解决 `resume_from_checkpoint` 在当前链路未按“从指定 step 接着跑”生效的问题
  - 使“基于既有 best checkpoint 继续训练”可直接执行

### 9.2 基于 `checkpoint-300` 的 2k 续训
- 命令核心参数：
  - `--init_model_dir data/ckpts/oneflow_text_pt_910c_baseline_s500/checkpoint-300`
  - `--max_steps 2000`
  - `--per_device_train_batch_size 16`
  - `--save_steps 400`
- 输出目录：
  - `data/ckpts/oneflow_text_pt_910c_continue_from300_s2000`
- 结果：
  - `exit_code=0`
  - `train_runtime=1414.8214s`
  - `train_steps_per_second=1.414`
  - `train_loss=17.0364`
  - 产物：`checkpoint-400/800/1200/1600/2000` + `checkpoint-final`

### 9.3 prompt + loss 联合 sweep（hybrid）
- 执行脚本：
  - `scripts/oneflow/eval_text_only_checkpoint_sweep.py`
- 关键设置：
  - `run_prompt_eval=True`
  - `ranking_mode=hybrid`
  - `loss_num_batches=30`
  - `loss_seeds=41,42,43`
- 输出：
  - `data/ckpts/oneflow_text_pt_910c_continue_from300_s2000/stable_eval_prompt/summary.md`
  - `data/ckpts/oneflow_text_pt_910c_continue_from300_s2000/stable_eval_prompt/summary.json`
- 排名结果：
  1. `checkpoint-1600` (`loss_tok_mean=10.1370`, `loss_total_mean=8.6357`, `prompt_pass_rate=0.0%`)
  2. `checkpoint-2000`
  3. `checkpoint-1200`
  4. `checkpoint-800`
  5. `checkpoint-400`
- 当前 best checkpoint（进入下一阶段）：
  - `checkpoint-1600`

### 9.4 解释与下一步
- 在本轮 minimal prompt 集下，所有 checkpoint 的 `expected_pass_rate` 均为 `0.0%`，说明该集合尚不能有效区分“更好的 PT checkpoint”。
- 因此下一轮应优先：
  - 扩充/分层 prompt 集；
  - 再跑一次 hybrid sweep（或 `prompt_first`）确认 prompt 维度是否具备判别力；
  - 并行推进 2 节点短程 bring-up。

---

## 10) Day 4-7 执行更新（进行中）

### 10.1 2 节点 bring-up 启动器落地
- 新增配置：`scripts/accelerate_configs/npu_ddp_2node.yaml`
- 新增脚本：`scripts/oneflow/launch_pt_text_910c_2node.sh`
- 功能：
  - 支持 `--num_machines/--num_processes/--machine_rank/--master_addr/--main_process_port`
  - 支持 `--init_model_dir`（可直接从 `checkpoint-1600` 等热启动）
  - 默认离线训练变量，兼容 `init_env.sh` / `activate_python_env.sh`

### 10.2 启动器链路验证（单机降级 smoke）
- 验证命令：`num_machines=1`, `num_processes=16`, `max_steps=1`
- 输出目录：`data/ckpts/oneflow_text_pt_910c_2node_launcher_smoke_s1`
- 结果：
  - `exit_code=0`
  - `train_runtime=5.1918s`
  - `train_steps_per_second=0.193`
  - `train_loss=3.6836`
- 结论：脚本本身可用，训练链路可起。

### 10.3 两节点实跑阻塞点（待集群侧补齐）
- 当前节点：`zjow-dev-oneflow-worker-0`（`10.119.10.155`）
- 现状：`zjow-dev-oneflow-worker-1` 当前不可解析，无法直接发起双节点同步启动。
- 需要补齐：
  - rank1 节点可达地址（IP/hostname）
  - 两节点同时执行命令（仅 `--machine_rank` 不同）

### 10.4 分层 prompt 集 + 新一轮 sweep
- 新增 prompts：`scripts/oneflow/eval_prompts_text_tiered_v1.jsonl`（16 条，easy/medium/hard 分桶）
- sweep 输出：
  - `data/ckpts/oneflow_text_pt_910c_continue_from300_s2000/stable_eval_prompt_tiered_v1/summary.md`
  - `data/ckpts/oneflow_text_pt_910c_continue_from300_s2000/stable_eval_prompt_tiered_v1/summary.json`
- 关键结果（`ranking_mode=hybrid`）：
  1. `checkpoint-400` (`prompt_pass_rate=43.75%`, `loss_tok_mean=33.95`)
  2. `checkpoint-2000` (`prompt_pass_rate=31.25%`, `loss_tok_mean=39.53`)
  3. `checkpoint-1600` (`prompt_pass_rate=25.00%`, `loss_tok_mean=9.999`)
  4. `checkpoint-800` (`prompt_pass_rate=18.75%`, `loss_tok_mean=19.23`)
  5. `checkpoint-1200` (`prompt_pass_rate=12.50%`, `loss_tok_mean=41.34`)
- 解释：
  - 与 minimal 集（全 0）相比，tiered 集已产生非零区分度；
  - 但当前 `hybrid`（prompt 优先）会把高 prompt/高 loss checkpoint 提前，存在“选点反转”风险；
  - hard 桶仍全 0，提示该集合还不够“能力导向”。

### 10.5 Day 6-7 当前决策（主配置）
- （原决策）训练续跑 checkpoint 曾暂定继续使用 `checkpoint-1600`。
- （当前更新）先在 11.8 单节点 1000-step 追加验证中更新为 `checkpoint-200`，再在 11.10 单节点复核中前移更新为 `checkpoint-100`，`checkpoint-1600` 继续作为回退点。
- 选点规则（临时）：
  - 主排序：`loss_tok`
  - 或采用 loss-guard（如 `loss_tok <= 2x best_loss_tok`）后再按 prompt 排序
- 在该 guard 下，历史 2k run 候选排序为：
  1. `checkpoint-1600`
  2. `checkpoint-800`
- 在本次单节点 debug 链路内，局部最优已由 `checkpoint-200` 前移到 `checkpoint-100`（见 11.10）。

### 10.6 下一步执行
- 拿到 rank1 可达地址后，直接用 `launch_pt_text_910c_2node.sh` 跑 200-500 steps bring-up（优先从 `checkpoint-100` 热启动，`checkpoint-1600` 作为回退，`checkpoint-200` 作为次级备选）。
- 并行迭代 tiered prompts（重点提升 medium/hard 判别力），再做一轮 guarded-hybrid 复核。

---

## 11) 单节点调试补充（当前阶段）

### 11.1 单节点续训 A（`lr=1e-4`，200 steps）
- 命令：`launch_pt_text_910c_2node.sh`（`num_machines=1`, `num_processes=16`）+ `--init_model_dir .../checkpoint-1600`
- 输出目录：`data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s200`
- 训练结果：
  - `exit_code=0`
  - `train_runtime=135.178s`
  - `train_steps_per_second=1.48`
  - `train_loss=15.5186`
- 评测（tiered prompts + loss）：
  - `checkpoint-200`: `prompt_pass_rate=31.25%`, `loss_tok_mean=21.7881`, `loss_total_mean=15.3204`

### 11.2 单节点续训 B（`lr=2e-5`，200 steps，对照）
- 输出目录：`data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s200_lr2e5`
- 训练结果：
  - `exit_code=0`
  - `train_runtime=134.5535s`
  - `train_steps_per_second=1.486`
  - `train_loss=15.5075`
- 对照评测（`checkpoint-200`）：
  - `loss_tok_mean=17.6782`（优于 `lr=1e-4` 的 `21.7881`）
  - `loss_total_mean=13.0483`（优于 `lr=1e-4` 的 `15.3204`）
  - `prompt_pass_rate=31.25%`（与 `lr=1e-4` 相同）

### 11.3 单节点续训 C（`lr=1e-5`，200 steps，对照）
- 输出目录：`data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s200_lr1e5`
- 训练结果：
  - `exit_code=0`
  - `train_runtime=135.9630s`
  - `train_steps_per_second=1.471`
  - `train_loss=15.5123`
- 对照评测（`checkpoint-200`）：
  - `loss_tok_mean=28.8904`（劣于 `lr=2e-5` 与 `lr=1e-4`）
  - `loss_total_mean=19.2134`
  - `prompt_pass_rate=31.25%`（与其余两组相同）

### 11.4 当前结论（单节点，三组对照）
- 从 `checkpoint-1600` 继续短程 200 steps 时，三组结果为：
  - `lr=2e-5`: `loss_tok_mean=17.6782`（最好）
  - `lr=1e-4`: `loss_tok_mean=21.7881`
  - `lr=1e-5`: `loss_tok_mean=28.8904`（最差）
- 三组 `prompt_pass_rate` 均为 `31.25%`，说明当前 tiered prompts 对这组短程 LR 差异不敏感，主要区分来自 loss。
- 因此当前建议：单节点调试优先 `lr=2e-5`，并保持 `loss_tok` 作为主选点指标；`lr=1e-5` 不建议继续。
- 另外，三组短程结果均未超过原 `checkpoint-1600` 的基线质量（此前 `loss_tok_mean≈9.999`），短程续训 checkpoint 不应直接替代主 baseline。

### 11.5 多节点脚本交付（由外部手动启动）
- 脚本：`scripts/oneflow/launch_pt_text_910c_2node.sh`
- 配置：`scripts/accelerate_configs/npu_ddp_2node.yaml`
- 启动方式：两节点执行同一命令，仅 `--machine_rank` 分别为 `0/1`。

### 11.6 单节点续训 D（`lr=2e-5`，500 steps，补充验证）
- 命令口径：`launch_pt_text_910c_2node.sh`（`num_machines=1`, `num_processes=16`）+ `--init_model_dir .../checkpoint-1600`
- 输出目录：`data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s500_lr2e5`
- 训练结果：
  - `exit_code=0`
  - `train_runtime=345.0255s`
  - `train_steps_per_second=1.449`
  - `train_loss=16.7519`
  - 产物：`checkpoint-100/200/300/400/500` + `checkpoint-final`
- sweep 输出（tiered prompts + loss）：
  - `data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s500_lr2e5/stable_eval_prompt_tiered_v1/summary.md`
  - `data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s500_lr2e5/stable_eval_prompt_tiered_v1/summary.json`
- 关键结果（按 step）：
  - `checkpoint-100`: `loss_tok_mean=20.3078`, `loss_total_mean=14.4749`, `prompt_pass_rate=31.25%`
  - `checkpoint-200`: `loss_tok_mean=16.3530`, `loss_total_mean=12.1828`, `prompt_pass_rate=31.25%`
  - `checkpoint-300`: `loss_tok_mean=14.0675`, `loss_total_mean=11.0821`, `prompt_pass_rate=31.25%`
  - `checkpoint-400`: `loss_tok_mean=17.8723`, `loss_total_mean=13.1069`, `prompt_pass_rate=31.25%`
  - `checkpoint-500`: `loss_tok_mean=11.7046`, `loss_total_mean=9.7193`, `prompt_pass_rate=31.25%`
- 结果判定：
  - `ranking_mode=hybrid` 下 best 为 `checkpoint-500`
  - 由于本轮 5 个 checkpoint 的 `prompt_pass_rate` 完全相同（均 `31.25%`），实际排序完全由 loss 决定

### 11.7 当前补充结论（单节点）
- 在 `lr=2e-5` 下继续到 500 steps，run 内最优已从 `checkpoint-200` 推进到 `checkpoint-500`，说明该学习率在当前口径下可稳定推进。
- 当前 tiered prompts 对这轮 100-500 step 的区分度不足（全点同分），应继续以 `loss_tok` 作为主选点指标，prompt 仅作辅助手段。
- 该结论仅针对 500-step 子区间成立；在后续 1000-step sweep（见 11.8）中已出现更优点 `checkpoint-200`，后续策略更新为“`checkpoint-200` 候选 + `checkpoint-1600` 回退”。

### 11.8 单节点续训 E（`lr=2e-5`，1000 steps，继续验证）
- 命令口径：`launch_pt_text_910c_2node.sh`（`num_machines=1`, `num_processes=16`）+ `--init_model_dir .../checkpoint-1600`
- 输出目录：`data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s1000_lr2e5`
- 训练结果：
  - `exit_code=0`
  - `train_runtime=710.6319s`
  - `train_steps_per_second=1.407`
  - `train_loss=16.8206`
  - 产物：`checkpoint-100/200/300/400/500/600/700/800/900/1000` + `checkpoint-final`
- sweep 输出（tiered prompts + loss）：
  - `data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s1000_lr2e5/stable_eval_prompt_tiered_v1/summary.md`
  - `data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s1000_lr2e5/stable_eval_prompt_tiered_v1/summary.json`
- 关键结果（节选）：
  - `checkpoint-200`: `loss_tok_mean=9.9181`, `loss_total_mean=8.2188`, `prompt_pass_rate=31.25%`（本轮 best）
  - `checkpoint-900`: `loss_tok_mean=11.3223`, `loss_total_mean=9.4037`, `prompt_pass_rate=12.5%`
  - `checkpoint-500`: `loss_tok_mean=14.6226`, `loss_total_mean=11.2540`, `prompt_pass_rate=12.5%`
  - `checkpoint-1000`: `loss_tok_mean=19.8415`, `loss_total_mean=14.1600`, `prompt_pass_rate=12.5%`
- 与主 baseline `checkpoint-1600` 对比：
  - `checkpoint-1600`: `loss_tok_mean=9.9990`, `loss_total_mean=8.5464`, `prompt_pass_rate=25.0%`
  - `checkpoint-200`（本轮 best）小幅更优：`loss_tok` 下降约 `0.81%`，`loss_total` 更低，prompt 通过率提升 `6.25` 个百分点（`4/16 -> 5/16`）。
- 风险备注：
  - 原始 `hybrid` 仍有 prompt 优先偏置：如 `checkpoint-400/700` 因较高 prompt 被排在更前，但 loss 显著劣化；需继续采用 `loss_tok` 主导或 loss-guard。

### 11.9 当前补充结论（更新）
- `lr=2e-5` 继续保持为当前单节点调试首选学习率。
- 主 checkpoint 策略更新为：`checkpoint-200`（候选主点）+ `checkpoint-1600`（回退点）。
- 由于 `checkpoint-200` 相对 `checkpoint-1600` 的优势幅度较小，下一轮仍应优先复核稳定性（同口径复评或续训后再 sweep）。

### 11.10 单节点续训 F（从 `checkpoint-200` 继续，`lr=2e-5`，500 steps，稳定性复核）
- 命令口径：`launch_pt_text_910c_2node.sh`（`num_machines=1`, `num_processes=16`）+ `--init_model_dir data/ckpts/oneflow_text_pt_910c_single_debug_from1600_s1000_lr2e5/checkpoint-200`
- 输出目录：`data/ckpts/oneflow_text_pt_910c_single_debug_from200_s500_lr2e5`
- 训练结果：
  - `exit_code=0`
  - `train_runtime=342.755s`
  - `train_steps_per_second=1.459`
  - `train_loss=16.6863`
  - 产物：`checkpoint-100/200/300/400/500` + `checkpoint-final`
- sweep 输出（tiered prompts + loss）：
  - `data/ckpts/oneflow_text_pt_910c_single_debug_from200_s500_lr2e5/stable_eval_prompt_tiered_v1/summary.md`
  - `data/ckpts/oneflow_text_pt_910c_single_debug_from200_s500_lr2e5/stable_eval_prompt_tiered_v1/summary.json`
- 关键结果（按 step）：
  - `checkpoint-100`: `loss_tok_mean=9.3210`, `loss_total_mean=8.0644`, `prompt_pass_rate=31.25%`（本轮 best）
  - `checkpoint-300`: `loss_tok_mean=16.4428`, `loss_total_mean=11.7120`, `prompt_pass_rate=31.25%`
  - `checkpoint-400`: `loss_tok_mean=19.5235`, `loss_total_mean=13.8554`, `prompt_pass_rate=31.25%`
  - `checkpoint-500`: `loss_tok_mean=21.3245`, `loss_total_mean=14.6501`, `prompt_pass_rate=31.25%`
  - `checkpoint-200`: `loss_tok_mean=24.6680`, `loss_total_mean=17.0765`, `prompt_pass_rate=31.25%`
- 与主 baseline `checkpoint-1600` 对比：
  - `checkpoint-1600`: `loss_tok_mean=9.9990`, `loss_total_mean=8.5464`, `prompt_pass_rate=25.0%`
  - `checkpoint-100`（本轮 best）：`loss_tok` 约 `-6.78%`，`loss_total` 约 `-5.64%`，prompt 通过率提升 `6.25` 个百分点（`4/16 -> 5/16`）。
- 与上一轮候选 `checkpoint-200`（11.8）对比：
  - `checkpoint-200`（11.8 best）: `loss_tok_mean=9.9181`, `loss_total_mean=8.2188`, `prompt_pass_rate=31.25%`
  - 本轮 `checkpoint-100` 进一步改善：`loss_tok` 约 `-6.02%`，`loss_total` 约 `-1.88%`，prompt 持平。

### 11.11 当前补充结论（再更新）
- 从“`checkpoint-1600 -> checkpoint-200`”继续做同口径复核后，最优点前移至 `checkpoint-100`，说明当前单节点 `lr=2e-5` 下存在“短程再优化窗口”。
- 当前主 checkpoint 策略更新为：`checkpoint-100`（候选主点）+ `checkpoint-1600`（回退点）+ `checkpoint-200`（次级备选）。
- 本轮 5 个 checkpoint 的 `prompt_pass_rate` 仍全部相同（`31.25%`），排序仍主要由 `loss_tok` 决定；后续应继续沿用 `loss_tok` 主导 / loss-guard 规则。
