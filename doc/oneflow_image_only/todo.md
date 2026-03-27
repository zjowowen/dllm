# oneflow_image_only TODO

## Completed

- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_image_only/` 已存在。
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_image_only/pt_image.py` 已存在。
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_image_only/eval_image_sample.py` 已存在。

## 现状确认

- [ ] 当前 worktree 未提供独立的 image-only 910C launcher；执行训练时复用现有 `accelerate launch` 模板并保持入口为 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_image_only/pt_image.py`。
- [ ] 当前 worktree 未提供独立的 image-only loss eval 脚本；如需离线 loss 检查，补充专项脚本或在训练日志中记录。
- [ ] 先跑一次 smoke：forward/backward 不崩、image loss 有限、采样脚本可读取 checkpoint。

## 基线实验（单 shard overfit）

- [ ] Run 5000 step overfit on flower32 single shard (shard-000005).
- [ ] Verify image loss 持续下降（无发散/NaN）.
- [ ] Run sampling + VAE decode on overfit checkpoint，验证重建质量.
- [ ] Record loss curve and reconstruction samples in PROGRESS.md.

## 扩展实验

- [ ] Run full flower32 dataset training (1000 steps).
- [ ] Compare image_loss_weight = {0.1, 1.0, 5.0, 10.0}.
- [ ] Use `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_image_only/eval_image_sample.py` 评估不同 `dt = {0.02, 0.05, 0.1}` 的采样质量。

## Risks to monitor

- [ ] Latent scale 一致性（训练 vs decode 必须用同一 scale）.
- [ ] image_num_tokens 必须与 latent H×W 匹配.
- [ ] τ_text > 1 时 text loss 不应干扰 image 优化.
- [ ] 910C 上 mixed-modal 序列的显存边界（text + image tokens 总长度）.
