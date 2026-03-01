# oneflow_text_only TODO

## Completed

- [x] Create `dllm/pipelines/oneflow_text_only` package scaffold and lazy exports.
- [x] Implement `OneFlowTextOnlyModel` (reference-style DDiT backbone) with OneFlow output contract.
- [x] Add text-only training entry `examples/oneflow_text_only/pt_text.py`.
- [x] Add visually_generate-style visualization module (`visualize.py`).
- [x] Add script entries:
  - [x] `scripts/oneflow_text_only/eval_text_only_prompts.py`
  - [x] `scripts/oneflow_text_only/eval_text_only_loss.py`
  - [x] `scripts/oneflow_text_only/launch_pt_text_910c.sh`
  - [x] `scripts/oneflow_text_only/launch_pt_text_910c_nnode.sh` (N-node, configurable)
  - [x] `scripts/oneflow_text_only/eval_text_only_checkpoint_sweep.py` (batch eval + ranking)
- [x] Run local smoke checks (compile + runtime + artifact export).

## Next recommended experiments (single-node debug)

- [ ] Run 20-50 step quick train on 910C with `oneflow_text_only`.
- [ ] Run tiered prompt eval with `--visualize True`.
- [ ] Run loss eval (`loss_total/loss_tok`) and compare against current `oneflow` baseline.
- [ ] Select first candidate checkpoint for continued debug (single-node only).

## Risks to monitor

- [ ] Ensure attention mask semantics remain stable under long sequences.
- [ ] Check whether sample-level time conditioning (`times.mean`) is sufficient for target behavior.
- [ ] Validate prompt quality trend against existing `oneflow` pipeline at equal budget.

