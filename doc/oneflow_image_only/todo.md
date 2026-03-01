# oneflow_image_only TODO

## Completed

（暂无）

## Pipeline 搭建

- [ ] Create `dllm/pipelines/oneflow_image_only` package scaffold and lazy exports.
- [ ] Implement `OneFlowImageOnlyTrainer`（继承 OneFlowTrainer，锁定 image-only 配置）.
- [ ] Add training entry `examples/oneflow_image_only/pt_image.py`.
- [ ] Add script entries:
  - [ ] `scripts/oneflow_image_only/launch_pt_image_910c.sh`
  - [ ] `scripts/oneflow_image_only/eval_image_only_loss.py`
  - [ ] `scripts/oneflow_image_only/eval_image_only_sample.py`
- [ ] Run local smoke checks (compile + runtime + loss finite).

## 基线实验（单 shard overfit）

- [ ] Run 5000 step overfit on flower32 single shard (shard-000005).
- [ ] Verify image loss 持续下降（无发散/NaN）.
- [ ] Run sampling + VAE decode on overfit checkpoint，验证重建质量.
- [ ] Record loss curve and reconstruction samples in PROGRESS.md.

## 扩展实验

- [ ] Run full flower32 dataset training (1000 steps).
- [ ] Compare image_loss_weight = {0.1, 1.0, 5.0, 10.0}.
- [ ] Evaluate sampling quality with different dt = {0.02, 0.05, 0.1}.

## Risks to monitor

- [ ] Latent scale 一致性（训练 vs decode 必须用同一 scale）.
- [ ] image_num_tokens 必须与 latent H×W 匹配.
- [ ] τ_text > 1 时 text loss 不应干扰 image 优化.
- [ ] 910C 上 mixed-modal 序列的显存边界（text + image tokens 总长度）.
