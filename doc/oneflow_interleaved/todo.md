# oneflow_interleaved TODO

## Completed

- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_interleaved/` 已存在。
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/pt_interleaved.py` 已存在。
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py` 已存在。

## 现状确认

- [ ] 当前 worktree 未提供 interleaved 专用 launcher；执行训练时复用现有 `accelerate launch` 模板并保持入口为 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/pt_interleaved.py`。
- [ ] 当前 worktree 未提供 dedicated interleaved loss / sample eval 脚本；不要把这些脚本写成已存在。
- [ ] 先跑一次 smoke：训练入口可起、`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_interleaved/sample_interleaved.py` 能读取 checkpoint，并确认 `bs=1` / `infill` 限制未被忽略。

## Phase 2b 实验（Interleaved 训练）

- [ ] Exp 2b-1: ctmc + τ_max=2.0 + weight=1.0 (5000 steps).
- [ ] Exp 2b-2: ctmc + τ_max=2.0 + dynamic weight (5000 steps).
- [ ] Exp 2b-3: paper + τ_max=2.0 + weight=1.0 (5000 steps).
- [ ] 统计图像删除率分布.
- [ ] 分析过渡区间 τ_text ∈ (0.9, 1.1) 的 loss 行为.

## Phase 3 实验（训练-推理闭环）

### 3a: Text-only 采样
- [ ] 从最优 checkpoint 采样纯文本.
- [ ] 验证序列增长和 π gate 行为.

### 3b: Image-conditioned 采样
- [ ] 给定 prompt 采样图像.
- [ ] VAE decode 并评估质量.

### 3c: Interleaved 采样
- [ ] 从 BOS 开始交错生成.
- [ ] 验证 <|oneflow_image|> 插入和图像 denoise.
- [ ] 验证终止条件.

## 一致性验证

- [ ] 统一序列构建：train vs sampler builder 输出 diff.
- [ ] condition_text_on_time 训练/推理对齐检查.
- [ ] image_num_tokens 一致性检查.
- [ ] 新图像创建逻辑端到端测试.

## Phase 4 消融（基于最优配置）

- [ ] text_loss_type: paper vs ctmc
- [ ] image_loss_weight: 0.1, 1.0, 5.0, 10.0
- [ ] condition_text_on_time: True vs False
- [ ] scheduler: linear vs cosine
- [ ] τ_text_max: 1.0 vs 2.0
- [ ] dt (推理): 0.02, 0.05, 0.1

## Risks to monitor

- [ ] τ_text 过渡区间 loss 异常.
- [ ] 多图连续删除后 bag 合并正确性.
- [ ] 推理时序列动态增长超限.
- [ ] 训练/推理构建不一致导致生成退化.
- [ ] w(t) 截断值对采样质量的影响.
