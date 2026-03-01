# OneFlow Interleaved Generation 训练进展

> **目标**：验证完整的 OneFlow interleaved schedule（τ_text ~ Unif[0,2]），实现文本与图像的交错生成训练和推理闭环。
>
> **对应实验阶段**：`doc/oneflow/text_image_interleaved_review_zh.md` → Phase 2b + Phase 3
>
> **前置依赖**：
> - Phase 1a/1b（单模态基线均已通过）
> - Phase 2a（混合模态训练验证 text + image loss 不互相干扰）

---

## 范围锁定

- **训练目标**：完整 Algorithm 3，τ_text ~ Unif[0,2]
  - τ_text ∈ [0,1]：文本 κ-keep noising + 图像 interleaved schedule（τ_img = τ_text - κ^{-1}(u)）
  - τ_text ∈ (1,2]：文本全保留，仅训练图像 flow matching
- **模型**：OneFlow Transfusion trunk + 四个 head（π, λ, Q, v）
- **推理**：完整 Algorithm 1-2，文本 Bernoulli 插入 + 图像 Euler 更新
- **训练-推理一致性**：本阶段的核心验证目标

---

## 实现状态

### Pipeline

- [ ] `dllm/pipelines/oneflow_interleaved/` pipeline scaffold
- [ ] `examples/oneflow_interleaved/pt_interleaved.py` 训练入口
- [ ] `examples/oneflow_interleaved/sample_interleaved.py` 采样入口
- [ ] `scripts/oneflow_interleaved/launch_pt_interleaved_910c.sh` 910C 启动脚本
- [ ] `scripts/oneflow_interleaved/eval_interleaved_loss.py` 分模态 loss 评测
- [ ] `scripts/oneflow_interleaved/eval_interleaved_sample.py` 交错采样评测

### Phase 2b 验证（Interleaved 训练）

- [ ] τ_text ~ Unif[0,2] 全范围训练不崩溃
- [ ] τ_text > 1 区间的纯图像训练正常
- [ ] τ_text ∈ (0.9, 1.1) 过渡区间无异常
- [ ] 图像删除率统计符合预期
- [ ] loss 曲线无异常（特别关注过渡区间）

### Phase 3 验证（训练-推理一致性闭环）

#### 3a: Text-only 采样
- [ ] 训练后模型能生成有意义的文本
- [ ] 序列长度随步数增长
- [ ] π gate 后期步数插入概率下降

#### 3b: Image-only 采样
- [ ] 给定文本 prompt，采样图像 latent
- [ ] VAE decode 后图像非纯噪声
- [ ] Overfit 模型重建结果与 GT 有语义关联

#### 3c: Interleaved 采样
- [ ] 从 BOS 开始，文本和图像交错生成
- [ ] 模型在合适位置插入 `<|oneflow_image|>`
- [ ] 图像 latent 在后续步骤中正确 denoise
- [ ] 终止条件正常触发
- [ ] 序列长度增长合理

### 一致性检查

- [ ] 训练/推理统一序列构建一致性（`build_unified_train_batch` vs `build_unified_sampler_inputs_bs1`）
- [ ] `condition_text_on_time` 训练/推理对齐
- [ ] `image_num_tokens` 训练/推理一致
- [ ] 新图像创建逻辑（推理中采样到 `<|oneflow_image|>` 时）

---

## 实验矩阵

### Phase 2b：Interleaved 训练

| 编号 | text_loss_type | τ_text_max | image_loss_weight | 数据 | 步数 | 状态 |
|------|---------------|-----------|------------------|------|------|------|
| 2b-1 | ctmc | 2.0 | 1.0 | flower32 + fineweb | 5000 | [ ] |
| 2b-2 | ctmc | 2.0 | 动态 | flower32 + fineweb | 5000 | [ ] |
| 2b-3 | paper | 2.0 | 1.0 | flower32 + fineweb | 5000 | [ ] |

### Phase 3：采样验证

| 编号 | 模式 | checkpoint 来源 | 步数 | dt | 状态 |
|------|------|----------------|------|-----|------|
| 3a-1 | text_only | 2b-1 best | 20 | 0.05 | [ ] |
| 3b-1 | image_conditioned | 2b-1 best | 20 | 0.05 | [ ] |
| 3c-1 | interleaved | 2b-1 best | 40 | 0.05 | [ ] |

---

## 关键配置

```yaml
# interleaved 训练核心参数
tau_text_max: 2.0                # 完整 interleaved range
mixed_generation_prob: 1.0       # 所有样本都包含图像
image_loss_weight: 1.0           # 待定（基于 Phase 2a 结论）
text_loss_type: ctmc             # 稳定文本 loss
condition_text_on_time: True     # 开启时间条件
normalize_text_loss_by_length: true
log_split_losses: true
```

```yaml
# interleaved 采样核心参数
dt: 0.05                         # Euler 步长
max_steps: 40                    # 最大步数
use_pi_gate: true                # 启用 π gate
max_new_tokens: 256              # 最大新 token 数
max_seq_len: 512                 # 最大序列长度
condition_text_on_time: true     # 与训练一致
```

---

## 关键观测指标

### 训练侧
1. loss_text 和 loss_img 的分时段趋势（τ_text ≤1 vs τ_text >1）
2. 图像删除率（P(τ_img < 0)）的统计分布
3. 梯度范数在过渡区间的行为
4. 总 loss 收敛趋势

### 推理侧
1. 文本序列增长曲线（每步插入 token 数）
2. 图像 latent 范数随步数的变化（应收敛）
3. `<|oneflow_image|>` 的插入时机和频率
4. 终止步数分布

### 一致性
1. 训练/推理统一序列 diff（固定输入）
2. π 分布在训练 vs 推理中的一致性
3. 推理插入率 vs 训练对应 t 的删除率

---

## 风险监控

- [ ] τ_text ∈ (0.9, 1.1) 过渡区间的 loss 异常
- [ ] 图像删除后 bag 合并的正确性（多图连续删除场景）
- [ ] 推理时序列动态增长超出 max_seq_len
- [ ] 推理时新图像创建过多导致 OOM
- [ ] 训练/推理统一序列构建不一致导致生成质量退化
- [ ] w(t) 在 t→1 时的截断（max_w）对采样率的影响

---

## 参考文档

- 架构审查（完整）：`doc/oneflow/text_image_interleaved_review_zh.md`
- 训练-推理一致性矩阵：同上 § 3
- Mixed generation 基线：`doc/oneflow_mixed_generation/PROGRESS.md`
- 论文 Algorithm 1-3：`doc/oneflow/design/oneflow_paper_spec_2510_03506.md`
- 归零验证 Stage 3-5：`doc/oneflow/validation/oneflow_zero_validation_zh.md`
