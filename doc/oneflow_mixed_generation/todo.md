# oneflow_mixed_generation TODO

## Completed

（暂无）

## Pipeline 搭建

- [ ] Create `dllm/pipelines/oneflow_mixed_generation` package scaffold and lazy exports.
- [ ] Implement `OneFlowMixedGenerationTrainer`（继承 OneFlowTrainer，设定 mixed generation 默认配置）.
- [ ] Add training entry `examples/oneflow_mixed_generation/pt_mixed.py`.
- [ ] Add script entries:
  - [ ] `scripts/oneflow_mixed_generation/launch_pt_mixed_910c.sh`
  - [ ] `scripts/oneflow_mixed_generation/eval_mixed_loss.py`
- [ ] Run local smoke checks (compile + runtime + both losses finite).

## 基线实验（Phase 2a）

- [ ] Exp 2a-1: ctmc + mixed_gen_prob=0.5 + weight=1.0 (2000 steps).
- [ ] Exp 2a-2: ctmc + mixed_gen_prob=0.5 + weight=5.0 (2000 steps).
- [ ] Exp 2a-3: ctmc + mixed_gen_prob=0.2 + weight=1.0 (2000 steps).
- [ ] Exp 2a-4: paper + mixed_gen_prob=0.5 + weight=1.0 (2000 steps).
- [ ] 对比分析：确定最优 mixed_gen_prob 和 image_loss_weight.

## 验证项

- [ ] text loss 和 image loss 均呈下降趋势.
- [ ] 梯度范数无持续发散.
- [ ] 确认 split loss 日志正常输出.
- [ ] 910C 显存测试：确定可行的 batch size 上限.

## Risks to monitor

- [ ] Text/image loss 量级差异导致梯度不平衡.
- [ ] mixed_generation_prob 过低时图像 loss 更新不充分.
- [ ] 多模态序列超出 910C 显存限制.
- [ ] 批次内 text-only 和 text+image 样本的 padding 低效.
