# 控制变量实验 TODO 列表 (Human Execution)

由于完整的 16 卡 910C 训练可能需要较长时间，并且需要分配集群资源，请用户（Human）按照以下步骤执行控制变量实验，以验证影响 OneFlow 训练稳定性的关键因素。

实验进展记录在 `dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS_experiment.md` 中。你可以把实验运行得到的 loss 截图或者终端输出填写进去。

## 准备工作

*   为了快速验证，建议在单机或多机环境下，将最大步数（`max_steps`）设置在一个较小的范围（如 100~300 步），重点观察训练**初期的 loss 规模**以及是否剧烈波动。
*   **注**：我已经将 Facebook 库中关键的 Rotary Position Embedding (RoPE) 的实现（包括预计算维度缓存和恒等变换 Value 的细节）迁移到了 OneFlow 结构中。现在的测试将建立在**核心结构细节对齐**的基础上。

你可以修改一个用于短时间测试的 launch script，或者直接在命令行传参。

---

## 实验 0：模型结构（RoPE）对齐验证
该实验旨在测试仅仅将 Rotary 机制对齐为 Facebook 版本，是否能缓解波动（目前代码中已默认包含该修改）。

### Exp 0A (Baseline with FB RoPE)
执行默认的 OneFlow 训练逻辑（`condition_text_on_time=False`, `text_loss_type=paper`），但此时的 RoPE 已经是 FB 版本。
**执行命令示例**：
```bash
bash scripts/oneflow_text_only/launch_pt_text_910c.sh \
  --pt_bundle data/offline/pt_text_fineweb_1024_100k \
  --output_dir data/ckpts/test_exp0a_baseline_fbrope \
  --max_steps 300 \
  --logging_steps 10
```
**期望结果/分析目标**：观察只对齐了 Rotary 位置编码，能否解决前期 loss 在 10~30 且剧烈波动的问题。如果依旧波动，说明并非 RoPE 引起。

---

## 实验 1：模型设计影响测试 (Time Conditioning)

该实验旨在测试**关闭时间条件**是否是导致 loss 异常的主要原因。保持 Loss 为默认的 `paper` (Eq7) 不变。

### Exp 1B (Time Cond Enabled)
在 Exp 0A 基础上，开启时间感知 (`condition_text_on_time=True`)。
**执行命令示例**：
```bash
bash scripts/oneflow_text_only/launch_pt_text_910c.sh \
  --pt_bundle data/offline/pt_text_fineweb_1024_100k \
  --output_dir data/ckpts/test_exp1b_timecond \
  --max_steps 300 \
  --logging_steps 10 \
  --condition_text_on_time True
```
**分析目标**：观察只开启时间条件后，loss 的均值和波动是否有所下降。

---

## 实验 2：Loss 设计影响测试 (Paper Eq7 vs CTMC)

该实验旨在测试**Loss函数的数学形式（是否带时间权重$w(t)$，是否受到Poisson项截断影响，归一化分母等）**是否是导致 loss 异常的主要原因。

### Exp 2B (CTMC Loss)
在 Exp 1B 的基础上，切换 Loss 类型为 `ctmc`，该模式包含时间权重且没有易导致数值不稳定的截断泊松项。
**执行命令示例**：
```bash
bash scripts/oneflow_text_only/launch_pt_text_910c.sh \
  --pt_bundle data/offline/pt_text_fineweb_1024_100k \
  --output_dir data/ckpts/test_exp2b_ctmc \
  --max_steps 300 \
  --logging_steps 10 \
  --condition_text_on_time True \
  --text_loss_type ctmc \
  --normalize_text_loss_by_length True
```
**分析目标**：观察使用更接近 Facebook Generalized KL 形式的 CTMC loss 后，初始 loss 是否回落到数学上合理的 10 左右（$\ln(|V|)$），并且曲线变得像 FB 一样平稳。

---

## 实验 3 (可选)：独立验证 Paper Eq7 的时间条件影响
如果你想交叉验证：
- `text_loss_type=ctmc` 且 `condition_text_on_time=False` (查看单纯依靠 CTMC Loss 能否稳定无时间条件的模型)。

## 执行后操作
请在测试完成后，将 0A, 1B, 2B 前 100 步的 log/loss 曲线截图或关键数据更新到 `PROGRESS_experiment.md` 的相应章节。如果 Exp 2B 能完美解决问题，则确认为前序报告中的两个推断均成立。