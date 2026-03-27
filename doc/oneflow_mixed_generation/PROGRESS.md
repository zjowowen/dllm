# OneFlow Mixed Generation 训练进展

> **目标**：验证文本 loss 与图像 loss 能同时优化且不互相干扰，通过 `mixed_generation_prob` 控制每个 batch 中包含图像的比例。
>
> **对应实验阶段**：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md` → Phase 2a
>
> **前置依赖**：
> - Phase 1a（text-only 基线：CTMC loss 在 Transfusion trunk 上复验，建议先完成对应验证）
> - Phase 1b（image-only 基线：flow matching overfit，建议先完成对应验证）

---

## 范围锁定

- **训练目标**：本阶段拟同时训练文本插入 loss + 图像 flow matching loss
- **入口层变量**：`mixed_generation_prob` 已在训练入口暴露；但当前不应表述为已证明的 core-path 样本混合开关
- **模型**：OneFlow Transfusion trunk + 四个 head（π, λ, Q, v），全部参与训练
- **文本 loss**：优先使用 CTMC loss（继承 text_only 发现）
- **图像 loss**：Flow Matching MSE（Eq. 9）
- **总 loss**：`L = L_text + image_loss_weight × L_img`
- **数据**：WDS latent shards（image-text pairs）+ 纯文本数据（按 prob 混合）

---

## 实现状态

### 入口/包装层现状

- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_mixed_generation/` 已存在；当前实现主要是对基础 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow` trainer/sampler 的轻包装
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_mixed_generation/pt_mixed.py` 已存在，可作为训练入口
- [ ] 当前目录下未找到 mixed-generation 专用 launcher 脚本
- [ ] 当前目录下未添加 dedicated mixed-generation 评测脚本
- [ ] 当前“已存在”仅代表 package / entry 已落位，不代表 mixed path 已完成实验验证

### 验证状态

- [ ] CPU smoke：text + image forward/backward 不崩
- [ ] 分 loss 日志：`log_split_losses=True`，确认 text/image loss 分别可见
- [ ] text loss 与 image loss 均呈下降趋势（不互相拉扯）
- [ ] 梯度范数无持续发散

### 实验矩阵（计划验证，尚未完成）

| 编号 | text_loss_type | mixed_gen_prob | image_loss_weight | 数据 | 步数 | 状态 |
|------|---------------|----------------|------------------|------|------|------|
| 2a-1 | ctmc | 0.5 | 1.0 | flower32 + fineweb | 2000 | [ ] |
| 2a-2 | ctmc | 0.5 | 5.0 | flower32 + fineweb | 2000 | [ ] |
| 2a-3 | ctmc | 0.2 | 1.0 | flower32 + fineweb | 2000 | [ ] |
| 2a-4 | paper | 0.5 | 1.0 | flower32 + fineweb | 2000 | [ ] |

---

## 关键配置

```yaml
# mixed generation 训练当前入口参数
mixed_generation_prob: 0.5       # 入口层暴露的样本混合比例
tau_text_max: 2.0                # 标准 τ_text 范围
image_loss_weight: 1.0           # 初始权重，待消融
text_loss_type: ctmc             # 稳定的文本 loss
condition_text_on_time: True     # 开启时间条件
normalize_text_loss_by_length: true
log_split_losses: true           # 分别记录 text/image loss
```

- 当前 mixed-generation package / sampler / trainer 主要复用基础 oneflow pipeline，不应写成已经形成独立验证过的混合训练主路径
- `mixed_generation_prob` 目前更接近实验入口层的控制参数；其是否真正稳定代表“样本级 text/image 混合开关”，仍需后续实验矩阵验证

---

## 关键观测指标

1. **loss_text** 与 **loss_img** 的独立趋势
2. 两种 loss 是否"拉扯"（一降一升）
3. 梯度范数稳定性
4. 不同 `mixed_generation_prob` 下的训练效率差异
5. `image_loss_weight` 对文本/图像质量 trade-off 的影响

---

## 风险监控

- [ ] Text loss 和 image loss 量级差异过大导致梯度主导
- [ ] `mixed_generation_prob` 过低导致图像 loss 更新不充分
- [ ] 多模态序列长度（text + image tokens）超出 910C 显存预算
- [ ] 图像样本的 batch padding 导致训练效率下降

---

## 参考文档

- 架构审查：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md` § 7 (Phase 2a)
- Text-only CTMC 基线：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_text_only/PROGRESS_experiment.md`
- Image-only 基线：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow_image_only/PROGRESS.md`
