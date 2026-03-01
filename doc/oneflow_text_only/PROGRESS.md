# OneFlow Text-Only Progress

## 0) Scope (locked)

- Training objective: keep existing OneFlow Eq(7) insertion-loss training flow, swap model backbone only.
- Visual scope: integrate visually_generate-style evaluation in `oneflow_text_only` only.
- Process records: maintain this file and `doc/oneflow_text_only/todo.md`.

## 1) Completed in this iteration

### 1.1 Pipeline skeleton

- Added package: `dllm/pipelines/oneflow_text_only/`
  - `__init__.py` (lazy import)
  - `trainer.py` (`OneFlowTextOnlyTrainer`, reusing OneFlowTrainer)
  - `sampler.py` (`OneFlowTextOnlySampler`, reusing OneFlowSampler)
  - `runtime_config.py` (re-export oneflow runtime config helpers)
  - `models/__init__.py`
- Updated top-level lazy export: `dllm/pipelines/__init__.py` includes `oneflow_text_only`.

### 1.2 Reference-style text backbone migration

- Added `dllm/pipelines/oneflow_text_only/models/text_only_model.py`:
  - `OneFlowTextOnlyConfig`
  - `OneFlowTextOnlyModel`
  - DDiT-style blocks (`_DDiTBlock`, timestep embedding, rotary embedding)
  - OneFlow-compatible output contract:
    - `pi`
    - `lambda_nonzero`
    - `q_logits`
    - `v`
    - `hidden_states`
  - Implemented:
    - `save_pretrained(...)`
    - `from_pretrained(...)`
    - `resize_token_embeddings(...)`
  - Config compatibility:
    - writes `oneflow_text_only_config.json`
    - also writes `oneflow_config.json` for existing tooling compatibility.

### 1.3 Training entry (text-only)

- Added `examples/oneflow_text_only/pt_text.py`:
  - keeps current data/tokenization/collator flow
  - swaps model to `OneFlowTextOnlyModel`
  - reuses trainer via `OneFlowTextOnlyTrainer`
  - supports warm-start via `--init_model_dir`
  - keeps offline-mode and 910C usage habits aligned with existing scripts.

### 1.4 Visual evaluation integration

- Added `dllm/pipelines/oneflow_text_only/visualize.py`:
  - timeline conversion from sampler histories:
    - `build_intermediates_from_histories(...)`
  - colorized timeline rendering (Rich):
    - `render_rich_timeline(...)`
  - artifact export:
    - `save_visual_artifacts(...)` -> `.html` and `.json`
  - JSON includes:
    - step/time_grid
    - token update order
    - per-step changed token counts
    - valid sequence lengths.

### 1.5 Script entries

- Added `scripts/oneflow_text_only/eval_text_only_prompts.py`
  - new `--visualize*` options
  - exports prompt report and visual artifacts per prompt
  - loads runtime sampler config (`oneflow_runtime_config.json`) with CLI override logic.
- Added `scripts/oneflow_text_only/eval_text_only_loss.py`
  - Eq(7) offline loss eval on text-only checkpoints
  - supports multi-seed aggregation.
- Added `scripts/oneflow_text_only/launch_pt_text_910c.sh`
  - single-node 910C launcher
  - offline env defaults
  - warm-start support.

## 2) Smoke validation (local)

### 2.1 Syntax/import smoke

- `python -m py_compile` passed for all newly added text-only files.
- `--help` entry smoke passed:
  - `python examples/oneflow_text_only/pt_text.py --help`
  - `python scripts/oneflow_text_only/eval_text_only_prompts.py --help`
  - `python scripts/oneflow_text_only/eval_text_only_loss.py --help`

### 2.2 Runtime smoke

- Ran a small runtime check script:
  - instantiate `OneFlowTextOnlyModel` with tiny config
  - one forward pass shape check
  - save and reload checkpoint
  - build visual timeline from synthetic `histories`
  - export visual artifacts
- Result:
  - `SMOKE_OK`
  - generated files:
    - `visual_rank0_sample0.html`
    - `visual_rank0_sample0.json`

## 3) Repro commands (quick)

### 3.1 Train (single node, 910C launcher)

```bash
bash scripts/oneflow_text_only/launch_pt_text_910c.sh \
  --pt_bundle data/offline/pt_text_fineweb_1024_100k \
  --output_dir data/ckpts/oneflow_text_only_pt_910c_e4000 \
  --num_train_epochs 4000 \
  --save_every_epochs 200
```

### 3.2 Prompt eval + visualization

```bash
python scripts/oneflow_text_only/eval_text_only_prompts.py \
  --model_dir data/ckpts/oneflow_text_only_pt_910c_s200/checkpoint-200 \
  --prompts_file scripts/oneflow/eval_prompts_text_tiered_v1.jsonl \
  --output_dir data/ckpts/oneflow_text_only_pt_910c_s200/prompt_eval_tiered \
  --visualize True \
  --visualize_save_html True \
  --visualize_save_json True \
  --visualize_terminal False
```

### 3.3 Eq(7) loss eval

```bash
python scripts/oneflow_text_only/eval_text_only_loss.py \
  --model_dir data/ckpts/oneflow_text_only_pt_910c_s200/checkpoint-200 \
  --dataset_dir data/offline/pt_text_fineweb_1024_100k/dataset \
  --batch_size 8 \
  --num_batches 50 \
  --output_json data/ckpts/oneflow_text_only_pt_910c_s200/loss_eval.json
```

## 4) Notes

- This iteration focuses on implementation + local smoke. No long-running 910C training job is launched in this pass.
- Existing `oneflow` pipeline is untouched; all new behavior is isolated under `oneflow_text_only`.

## 5) Runtime hotfix (2026-02-18)

- During first real 910C launch, training crashed with:
  - `AttributeError: 'TrainingArguments' object has no attribute 'scheduler_cls'`
- Root cause:
  - `examples/oneflow_text_only/pt_text.py` missed `scheduler_cls` in `TrainingArguments`,
    while trainer construction expected it.
- Fix:
  - Added `scheduler_cls: str = "LinearKappaScheduler"` to `TrainingArguments`.
  - Verified by:
    - `python -m py_compile examples/oneflow_text_only/pt_text.py`
    - `python examples/oneflow_text_only/pt_text.py --help`
- Launcher update:
  - `scripts/oneflow_text_only/launch_pt_text_910c.sh` now defaults to epoch mode:
    - `num_train_epochs=4000`
    - `save_every_epochs=200`
  - `save_steps` is auto-computed from dataset size and global batch, so checkpoint spacing follows epoch intent.

## 6) N-node launcher (2026-02-24)

- Added: `scripts/oneflow_text_only/launch_pt_text_910c_nnode.sh`
- Goal:
  - Keep single-node launcher unchanged.
  - Provide a dedicated N-node launcher that scales by `--num_machines`.
- Key behaviors:
  - multi-node args:
    - `--num_machines`
    - `--machine_rank`
    - `--master_addr`
    - `--main_process_port`
    - `--processes_per_machine` / `--num_processes`
  - if `--num_processes` is not provided, auto-derive:
    - `num_processes = num_machines * processes_per_machine`
  - epoch-save policy is preserved:
    - defaults `num_train_epochs=4000`, `save_every_epochs=200`
    - auto-compute `save_steps` from dataset size and global batch
  - wandb behavior aligns with single-node script:
    - `WANDB_MODE=online` -> default `report_to=wandb`
    - otherwise -> `report_to=none`
  - supports pass-through extra HF args after `--`.

### 6.1 Example (2 nodes)

```bash
# node rank0
WANDB_MODE=online bash scripts/oneflow_text_only/launch_pt_text_910c_nnode.sh \
  --pt_bundle data/offline/pt_text_fineweb_1024_100k \
  --output_dir data/ckpts/oneflow_text_only_pt_910c_n2_e4000 \
  --num_machines 2 \
  --machine_rank 0 \
  --master_addr 10.119.10.155 \
  --num_train_epochs 4000 \
  --save_every_epochs 200

# node rank1 (same command, only rank differs)
WANDB_MODE=online bash scripts/oneflow_text_only/launch_pt_text_910c_nnode.sh \
  --pt_bundle data/offline/pt_text_fineweb_1024_100k \
  --output_dir data/ckpts/oneflow_text_only_pt_910c_n2_e4000 \
  --num_machines 2 \
  --machine_rank 1 \
  --master_addr 10.119.10.155 \
  --num_train_epochs 4000 \
  --save_every_epochs 200
```

## 7) Text-only checkpoint sweep entry (2026-02-26)

- Added: `scripts/oneflow_text_only/eval_text_only_checkpoint_sweep.py`
- Purpose:
  - evaluate all checkpoints under one root in batch
  - run loss eval + prompt eval per checkpoint
  - auto-rank and export summary
- Outputs:
  - `<output_dir>/summary.json`
  - `<output_dir>/summary.md`
  - per-checkpoint subdirs with:
    - `loss_report.json`
    - `prompt_eval/report.jsonl` (+ optional visual artifacts)
- Ranking modes:
  - `hybrid`
  - `loss_tok`
  - `loss_total`
  - `prompt_first`

### 7.1 Quick run example

```bash
python -u scripts/oneflow_text_only/eval_text_only_checkpoint_sweep.py \
  --ckpt_root data/ckpts/oneflow_text_only_ctmc_e400 \
  --dataset_dir data/offline/pt_text_fineweb_1024_100k/dataset \
  --prompts_file scripts/oneflow/eval_prompts_text_tiered_v1.jsonl \
  --output_dir data/ckpts/oneflow_text_only_ctmc_e400/stable_eval_tiered_v1 \
  --device npu \
  --loss_batch_size 8 \
  --loss_num_batches 100 \
  --loss_seeds 41,42,43 \
  --run_prompt_eval True \
  --prompt_device npu \
  --ranking_mode hybrid
```

## 8) Sweep runtime hotfix (2026-02-27)

- Symptom:
  - sweep failed at first checkpoint on NPU with
    `aclnnBinaryCrossEntropy` / `Cannot find bin of op BinaryCrossEntropy ... bf16`
- Root cause:
  - `scripts/oneflow_text_only/eval_text_only_loss.py` originally forced Eq(7) path
    (BCE on `pi`) regardless of training objective.
  - Current run used `text_loss_type=ctmc`, and NPU bf16 BCE kernel combination is not available.
- Fixes:
  1. `eval_text_only_loss.py` now resolves and respects `text_loss_type` from runtime config
     (`paper` or `ctmc`), with CLI override support.
  2. `paper` path is executed in fp32 for stability/kernel compatibility
     (`q_logits/pi/lam` cast to fp32 in eval computation).
  3. `dllm/pipelines/oneflow/losses.py` BCE in Eq(7) functions now runs in fp32.
- Validation:
  - one-checkpoint smoke eval on NPU passed:
    - `--model_dir data/ckpts/oneflow_text_only_ctmc_e400/checkpoint-5320`
    - `--batch_size 1 --num_batches 1`

