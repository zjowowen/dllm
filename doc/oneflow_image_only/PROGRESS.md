# OneFlow Image-Only 训练进展

> **目标**：孤立验证图像侧 Flow Matching 训练的正确性，不受文本插入 loss 干扰。
>
> **对应实验阶段**：`doc/oneflow/text_image_interleaved_review_zh.md` → Phase 1b
>
> **前置依赖**：Phase 0 归零验证已通过（Stage 0 latent decode + Stage 2 image flow matching 单测）

---

## 范围锁定

- **训练目标**：仅训练图像 flow matching loss（Eq. 9），文本侧处于"全保留"状态（τ_text 固定 > 1，或 t_text = 1）
- **模型**：复用 OneFlow 的 Transfusion trunk + 四个 head，但实际只优化 image head `to_v`
- **文本处理**：文本 token 全部保留（κ(1)=1），text loss 仅含 π BCE（推动 π→1），不影响图像训练
- **数据**：使用 WDS latent shards（预编码图像 latent + caption token ids）
- **评估**：
  - 训练侧：image loss 下降曲线
  - 推理侧：采样后 VAE decode 的图像质量（overfit 场景为 reconstruction）

---

## 实现状态

### Pipeline

- [ ] `dllm/pipelines/oneflow_image_only/` pipeline scaffold（trainer, sampler, runtime_config）
- [ ] `examples/oneflow_image_only/pt_image.py` 训练入口
- [ ] `scripts/oneflow_image_only/launch_pt_image_910c.sh` 910C 启动脚本
- [ ] `scripts/oneflow_image_only/eval_image_only_loss.py` 图像 loss 评测
- [ ] `scripts/oneflow_image_only/eval_image_only_sample.py` 采样 + VAE decode 评测

### 验证

- [ ] CPU smoke：forward + backward 不崩，image loss 有限
- [ ] Oracle 测试：当 v = flow_target 时 loss ≈ 0
- [ ] 单 shard overfit（5000 步）：image loss 持续下降
- [ ] Overfit reconstruction：采样结果与 GT 有语义关联

### 910C 实验

- [ ] 单 shard overfit（flower32 shard-000005）
- [ ] 全量 flower32 训练（1000 步）
- [ ] 采样 + VAE decode 可视化

---

## 关键配置

```yaml
# image-only 训练的核心参数
mixed_generation_prob: 1.0       # 所有样本都包含图像
tau_text_min: 1.5                # 固定 τ_text > 1，确保文本全保留
tau_text_max: 2.0                # τ_text ∈ [1.5, 2.0]
image_loss_weight: 1.0           # 图像 loss 权重
text_loss_type: ctmc             # 文本 loss 类型（此阶段文本 loss 极小）
condition_text_on_time: False    # 文本不条件于时间
```

---

## 风险监控

- [ ] Latent scale 一致性（训练用的 latent_scale 与 VAE decode 时必须一致）
- [ ] 图像 token 数量（`image_num_tokens`）与实际 latent shape 匹配
- [ ] 当 τ_text > 1 时 w(t_text) 的截断值（max_w）是否合理
- [ ] Overfit 场景下 image loss 能否降到接近 0

---

## 参考文档

- 论文图像 Flow Matching：`doc/oneflow/design/oneflow_paper_spec_2510_03506.md` § 2
- 归零验证 Stage 2：`doc/oneflow/validation/oneflow_zero_validation_zh.md` § Stage 2
- 架构审查：`doc/oneflow/text_image_interleaved_review_zh.md` § 1.3 (图像 loss) + § 6.2 (Phase 1b)
