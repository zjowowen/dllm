# OneFlow Doc Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align all OneFlow documentation with the current code and experimental reality so the repo clearly distinguishes implemented behavior, validated results, and planned work.

**Architecture:** Treat the documentation as a layered system. First update the top-level status sources so they become the canonical truth, then align per-track progress and todo files to that truth, and finally sweep supporting design and validation docs for stale claims or misleading command examples. Verification relies on targeted `grep` checks and manual readback of the edited Markdown.

**Tech Stack:** Markdown documentation, `grep`, `read`, `apply_patch`, git working tree review.

---

## File Map

**Canonical status sources**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md`

**Per-track progress docs**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS.md`

**Supporting design / validation / guide docs**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/data_readiness_zh.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_en.md`

**Track todo docs**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo_human.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo_human.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo_human.md`

**Verification inputs**
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/trainer.py`
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sequence_ops.py`
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_image_only/trainer.py`
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_mixed_generation/trainer.py`
- Read-only reference: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_interleaved/trainer.py`

### Task 1: Canonical Status Docs

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md`

- [ ] **Step 1: Prove the current top-level mismatch still exists**

Run:

```bash
grep -n "mixed_generation_prob" /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md
```

Expected: a line matching the old claim that mixed generation is already controllable in the core path.

- [ ] **Step 2: Rewrite `/doc/oneflow/PROGRESS.md` as the canonical truth source**

Replace the opening status bullets with wording in this shape:

```md
## 当前进展（按“代码已实现 / 已验证 / 待验证”划分）
- **基础 OneFlow 主干已实现**：核心训练、统一序列构造、interleaved image schedule、文本 loss 与图像 flow matching 都已在主干代码中存在。
- **text-only 是当前验证最完整的路线**：已有 smoke、baseline、checkpoint sweep、续训与多机 bring-up 记录。
- **image-only / mixed-generation / interleaved 已有入口和包装层**：对应 trainer、sampler、runtime config、examples 与部分脚本已存在，但实验闭环仍待补齐。
- **若干实验控制项仍停留在入口或文档语义层**：例如 `mixed_generation_prob` 与 image-only 的 `tau_text_min` 还不能等价视为“主干逻辑已经完整支持”。
- **采样器仍有 v1 边界**：当前仅支持 `bs=1`，`infill` 未实现。
```

Also replace the old “下一步任务” ordering with a reality-based priority list that starts with image-only validation, then mixed-generation control clarification, then interleaved closure.

- [ ] **Step 3: Simplify `/doc/oneflow/README.md` so it points to the new truth source**

Edit the quick-entry section to emphasize current-state reading order:

```md
## 快速入口
- 当前状态总览：`doc/oneflow/PROGRESS.md`
- 实验最成熟路线：`doc/oneflow_text_only/PROGRESS.md`
- 多模态路线现状：`doc/oneflow_image_only/PROGRESS.md`、`doc/oneflow_mixed_generation/PROGRESS.md`、`doc/oneflow_interleaved/PROGRESS.md`
- 设计与审查：`doc/oneflow/design/oneflow_design_zh.md`、`doc/oneflow/text_image_interleaved_review_zh.md`
```

Keep the file short; do not duplicate detailed status there.

- [ ] **Step 4: Verify the old top-level claim is gone**

Run:

```bash
grep -n "混合生成采样策略可控\|mixed_generation_prob" /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md
```

Expected: no line that claims the base pipeline already provides a fully controllable mixed-generation switch.

### Task 2: Per-Track Progress Docs

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS.md`

- [ ] **Step 1: Confirm the stale checklist language in the track docs**

Run:

```bash
grep -n "pipeline scaffold\|tau_text_min\|mixed_generation_prob" \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md
```

Expected: hits showing pipeline scaffold checklists and the `tau_text_min` / `mixed_generation_prob` assumptions.

- [ ] **Step 2: Rewrite `oneflow_image_only/PROGRESS.md` around current reality**

Replace the implementation-status block with a structure like:

```md
## 当前状态
- **代码现状**：`dllm/pipelines/oneflow_image_only/`、`examples/oneflow_image_only/pt_image.py`、`scripts/oneflow_image_only/launch_pt_image_910c.sh` 与 `scripts/oneflow_image_only/eval_image_sample.py` 已存在；当前更多是复用基础 OneFlow 主干的 image-only 包装层。
- **未落地项**：文档中的 `tau_text_min` 目前不是主干核心配置；image-only 仍主要依赖 `tau_text_max` 与 `t_text=min(1, tau_text)` 的现有行为。
- **实验现状**：尚未在文档中沉淀 overfit / reconstruction / decode 闭环结果。
```

Also replace the checklist items that say “pipeline scaffold” or reference missing `eval_image_only_loss.py` / `eval_image_only_sample.py` with wording that matches the real files.

- [ ] **Step 3: Rewrite `oneflow_mixed_generation/PROGRESS.md` to separate wrapper existence from mixed-control truth**

Use wording in this shape:

```md
## 当前状态
- **代码现状**：`dllm/pipelines/oneflow_mixed_generation/` 与 `examples/oneflow_mixed_generation/pt_mixed.py` 已存在，但 trainer/sampler 主要仍是对基础 `OneFlowTrainer` / `OneFlowSampler` 的默认值封装。
- **关键限制**：`mixed_generation_prob` 已在入口层出现，但当前文档不应把它表述成“主干训练逻辑中已经严格实现的样本混合开关”。
- **实验现状**：尚无稳定记录表明 text loss 与 image loss 的联合训练已经完成闭环验证。
```

Keep the experiment matrix, but mark it as planned validation rather than completed capability.

- [ ] **Step 4: Rewrite `oneflow_interleaved/PROGRESS.md` to emphasize dependence on the base pipeline**

Edit the implementation and consistency sections so they explicitly say:

```md
- `dllm/pipelines/oneflow_interleaved/` 与 `examples/oneflow_interleaved/*` 已存在，但当前核心计算仍回到基础 `dllm/pipelines/oneflow/` 主干。
- 采样能力仍受基础 `OneFlowSampler` v1 边界约束：仅 `bs=1`，`infill` 未实现。
- Phase 2b / Phase 3 的大部分条目仍属于待验证实验矩阵，而非已经完成的闭环。
```

Update any checklist item that still claims missing scaffold files when those files already exist.

- [ ] **Step 5: Verify `oneflow_text_only/PROGRESS.md` only if needed**

Run:

```bash
grep -n "mixed_generation_prob\|tau_text_min\|pipeline scaffold" /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS.md
```

Expected: no relevant hits; if the command returns no lines, leave the file unchanged.

### Task 3: Supporting Design, Validation, and Guide Docs

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/data_readiness_zh.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_en.md`

- [ ] **Step 1: Patch the review doc so methods are described as experimental intent, not current fact**

In `/doc/oneflow/text_image_interleaved_review_zh.md`, adjust the Phase 1b / 2a text to use wording like:

```md
**方法（目标形态）**：image-only 理想上应固定文本为全保留状态，只验证图像 flow matching；当前代码库已提供 image-only 包装层与训练入口，但 `tau_text_min` 这一独立核心配置仍未在主干逻辑中完整落地。

**方法（实验语义）**：mixed-generation 文档继续用 `mixed_generation_prob` 描述目标实验变量，但当前是否真正等价为主干样本混合开关，仍需以后续实现/验证为准。
```

Do not remove the experiment matrix; only correct the factual framing.

- [ ] **Step 2: Patch `data_readiness_zh.md` so it stops overstating runtime image mixing**

Replace the Phase 2a bullets with wording like:

```md
- 当前相关入口脚本提供 `--mixed_generation_prob` 参数，用于表达目标实验语义。
- 但文档不再假定基础主干已经严格以该参数实现“运行时 mask 掉图像”的完整混合逻辑；这一点应以后续实现与验证结果为准。
```

- [ ] **Step 3: Patch `oneflow_zero_validation_zh.md` command templates that use unsupported base-entry flags**

Update Templates C and D so they either:

```md
- 改用真正支持对应语义的专用入口（如 `examples/oneflow_image_only/pt_image.py`）；或
- 明确标注这些命令是“历史示例/目标实验语义”，不代表当前基础 `examples/oneflow/pt_wds_latents.py` 已接入 `mixed_generation_prob`。
```

The safer recommendation is to replace the commands with track-specific entrypoints rather than keep misleading flags on the base entry.

- [ ] **Step 4: Fix the one clear mismatch in `oneflow_design_en.md`**

If line 30 still says:

```md
- Sample `τ_text` in [0,1] (or [1,2] with `mixed_generation_prob`)
```

replace it with:

```md
- Sample `τ_text` in `[0, tau_text_max]`, then set `t_text=min(1, τ_text)`.
- The mixed / image-only regimes are documented as experimental stages; they should not be read as proof that every stage-specific control has already been wired into the base implementation.
```

- [ ] **Step 5: Verify the corrected support-layer wording**

Run:

```bash
grep -n "mixed_generation_prob\|tau_text_min" \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/data_readiness_zh.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_en.md
```

Expected: remaining hits, if any, should describe these terms as goals, entry-level semantics, or pending validation rather than established core behavior.

### Task 4: Track Todo Files

**Files:**
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo.md`
- Modify: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo_human.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo_human.md`
- Review-only unless mismatch found: `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo_human.md`

- [ ] **Step 1: Convert stale scaffold todos into reality-based execution lists**

For each machine-readable `todo.md`, replace opening sections like:

```md
- [ ] Create `dllm/pipelines/oneflow_image_only` package scaffold and lazy exports.
```

with status-aware text like:

```md
## Current code base
- [x] `dllm/pipelines/oneflow_image_only/` package exists.
- [x] `examples/oneflow_image_only/pt_image.py` exists.
- [x] launcher scripts exist.
- [ ] Real overfit / evaluation evidence is still missing from `PROGRESS.md`.
```

Apply the same pattern to the mixed-generation and interleaved todo files.

- [ ] **Step 2: Correct wrong script names in the todo files**

Replace references to missing files such as:

```md
scripts/oneflow_image_only/eval_image_only_loss.py
scripts/oneflow_image_only/eval_image_only_sample.py
scripts/oneflow_mixed_generation/eval_mixed_loss.py
scripts/oneflow_interleaved/eval_interleaved_loss.py
scripts/oneflow_interleaved/eval_interleaved_sample.py
```

with either real existing files or explicit “not yet added” notes. For image-only, use the real script name:

```md
scripts/oneflow_image_only/eval_image_sample.py
```

For mixed-generation and interleaved, prefer wording like “add dedicated eval script” instead of naming a file that does not yet exist.

- [ ] **Step 3: Review `todo_human.md` files for misleading implementation claims**

Run:

```bash
grep -n "mixed_generation_prob\|tau_text_min\|eval_.*sample\|eval_.*loss" \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo_human.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo_human.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo_human.md
```

If a line names a real command on a real file, keep it. If it implies nonexistent files or unsupported semantics, rewrite it as a human experiment note instead of a hard implementation fact.

- [ ] **Step 4: Verify no stale scaffold lines remain in the machine-readable todo files**

Run:

```bash
grep -n "Create `dllm/pipelines/oneflow_.*scaffold\|pipeline scaffold" \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo.md
```

Expected: no lines treating already-existing packages as not yet scaffolded.

### Task 5: Final Consistency Sweep

**Files:**
- Modify as needed: all files touched in Tasks 1-4

- [ ] **Step 1: Run a repo-wide terminology sweep**

Run:

```bash
grep -RIn "tau_text_min\|mixed_generation_prob\|only supports bs=1\|infill" /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc
```

Expected: every remaining hit should now be one of these:
- an accurate historical record,
- a stated future experiment variable,
- or an accurate statement of current sampler limitations.

- [ ] **Step 2: Read back the edited canonical docs for narrative consistency**

Read these files in full and verify they tell the same story:

```text
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md
/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md
```

Expected: all five files consistently state that text-only is the most validated path, while image-only / mixed-generation / interleaved are implemented as wrappers and entrypoints with validation still pending.

- [ ] **Step 3: Review git diff for scope discipline**

Run:

```bash
git diff -- /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers
```

Expected: diff is limited to Markdown wording, status tables, command examples, and doc structure; no code files are changed.

- [ ] **Step 4: Commit**

```bash
git add \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/README.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/PROGRESS.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/PROGRESS.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/PROGRESS.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/guides/data_readiness_zh.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_design_en.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/todo.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_mixed_generation/todo.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_interleaved/todo.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/specs/2026-03-27-oneflow-doc-alignment-design.md \
  /mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/docs/superpowers/plans/2026-03-27-oneflow-doc-alignment.md
git commit -m "docs: align oneflow progress with implementation reality"
```

Expected: one documentation-only commit.
