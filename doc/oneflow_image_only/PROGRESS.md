# OneFlow Image-Only 训练进展

> **目标**：孤立验证图像侧 Flow Matching 训练的正确性，不受文本插入 loss 干扰。
>
> **对应实验阶段**：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md` → Phase 1b
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

### 入口/包装层现状

- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow_image_only/` 已存在；当前仍主要是对基础 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow` 训练/采样逻辑的轻包装
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/examples/oneflow_image_only/pt_image.py` 已存在，可作为 image-only 训练入口
- [x] `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/scripts/oneflow_image_only/eval_image_sample.py` 已存在，可作为采样 + VAE decode 评测入口
- [ ] 当前目录下未找到独立 910C launcher 脚本
- [ ] 当前目录下未找到独立 image loss 评测脚本

### 验证状态

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
# image-only 训练当前可确认的核心参数
mixed_generation_prob: 1.0       # 当前入口层默认所有样本都带图像
tau_text_max: 2.0                # 当前核心逻辑主要依赖 tau_text_max
image_loss_weight: 1.0           # 图像 loss 权重
text_loss_type: ctmc             # 文本 loss 仍会走基础 oneflow 路径
condition_text_on_time: False    # 文本不条件于时间
```

- `tau_text_min` 不是当前 core config 的已确认字段；image-only 语义目前主要依赖 `tau_text_max` 与 `t_text = min(1, tau_text)` 后文本全保留这一基础路径
- 因此，这里的 image-only 更接近基于现有 oneflow 训练流的配置收缩，不应写成已独立闭环的新训练体系
- overfit / reconstruction / decode loop 仍应视为待验证，不应写成已闭环

---

## 风险监控

- [ ] Latent scale 一致性（训练用的 latent_scale 与 VAE decode 时必须一致）
- [ ] 图像 token 数量（`image_num_tokens`）与实际 latent shape 匹配
- [ ] 当 τ_text > 1 时 w(t_text) 的截断值（max_w）是否合理
- [ ] Overfit 场景下 image loss 能否降到接近 0

---

## 参考文档

- 论文图像 Flow Matching：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/design/oneflow_paper_spec_2510_03506.md` § 2
- 归零验证 Stage 2：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/validation/oneflow_zero_validation_zh.md` § Stage 2
- 架构审查：`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/doc/oneflow/text_image_interleaved_review_zh.md` § 1.3 (图像 loss) + § 6.2 (Phase 1b)
