# dLLM 依赖对齐设计

## 背景

当前 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml` 已将 `lm_eval` 写入默认依赖，但在实际环境中引入了一个依赖树冲突：

- 当前环境安装的是 `omegaconf==2.3.0`；
- 该版本要求 `antlr4-python3-runtime==4.9.*`；
- 而 `lm_eval` 的 `math` 相关 extra 需要 `antlr4-python3-runtime==4.11`；
- 实际安装后 `antlr4` 被提升到 `4.11.0`，导致 `omegaconf 2.3.0` 的约束被打破。

这说明当前问题不是“项目必须使用 `omegaconf==2.3.0`”，而是“项目当前未给出足够明确的依赖约束，导致 resolver 解出一个自相矛盾的组合”。

同时，`dllm/core/eval/base.py` 与测试导入链确实需要 `lm_eval`，因此简单删除它并不能解决问题；真正要解决的是：如何让项目的默认依赖策略与 `lm_eval`、`omegaconf`、`antlr4` 的组合保持一致。

## 目标

- 让 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml` 中的依赖关系不再自相矛盾；
- 优先支持 `lm_eval` 的可用性；
- 尽量避免把 `math` extra 带来的 `antlr4==4.11` 冲突无条件扩散到整个默认环境；
- 让新的开发环境在安装时就能得到一致的依赖解，而不是依赖手工 `pip install` 补洞。

## 非目标

- 不在本次工作中重构评测框架；
- 不修改 `dllm` 运行时代码以绕过 `lm_eval`；
- 不保证一次性解决所有第三方依赖版本问题；
- 不把 `sampler.py` 的兼容性 import 修补和依赖策略耦合为同一个设计问题。

## 已确认事实

### 1. `lm_eval` 是真实依赖

以下位置直接导入 `lm_eval`：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/core/eval/base.py`
- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/core/eval/mdlm.py`
- 多个 `dllm/pipelines/*/eval.py`

因此把 `lm_eval` 看成纯可有可无的临时依赖是不准确的。

### 2. 当前冲突来自已安装版本，而非项目显式 pin

`/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml` 中当前只写了：

- `omegaconf`
- `lm_eval`

并没有 pin `omegaconf==2.3.0`。也就是说，`2.3.0` 只是当前环境被解析出来的结果，不是项目必须坚持的目标版本。

### 3. 冲突点集中在 `math` extra

当前观察到的约束是：

- `omegaconf 2.3.0 -> antlr4-python3-runtime==4.9.*`
- `lm_eval[math] -> antlr4-python3-runtime==4.11`

因此冲突的核心不是 `lm_eval` 本体，而是 `lm_eval` 的数学评测额外依赖链。

## 方案比较

### 方案 A：升级 `omegaconf`（推荐）

思路：

- 先确认是否存在更新版本的 `omegaconf`，其依赖约束已经兼容 `antlr4 4.11`；
- 如果存在，则在 `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml` 中明确抬高 `omegaconf` 版本下界；
- 让默认环境能同时容纳 `omegaconf` 与 `lm_eval[math]`。

优点：

- 依赖图最干净；
- 默认安装即可获得完整评测能力；
- 不需要把 `lm_eval` 再拆成额外安装路径。

风险：

- 新版 `omegaconf` 是否与当前代码兼容还需验证；
- 如果其上游仍钉死 `antlr4 4.9.*`，该方案不可行。

### 方案 B：保留默认 `lm_eval`，拆出 `math` extra

思路：

- 默认依赖只保留基础 `lm_eval`；
- 将 `math_verify` / `antlr4 4.11` 一类仅服务于数学评测的链路拆入 optional extra；
- 主安装保持与 `omegaconf` 当前生态更稳定的组合。

优点：

- 对主环境最稳；
- 把冲突范围限制在真正需要数学评测的人身上；
- 更符合“默认安装提供核心功能，额外评测按需启用”的依赖分层思路。

风险：

- 默认环境的评测能力不再等同于 `lm_eval[math]` 全量能力；
- README 和安装文档需要同步说明。

### 方案 C：优先 `lm_eval[math]`，接受覆盖现有 `omegaconf` 约束

思路：

- 不主动调整 `omegaconf` 策略；
- 直接接受 `antlr4` 被抬到 `4.11`；
- 认为 resolver warning 可接受，只要运行时没立即坏即可。

优点：

- 最快；
- 对数学评测功能最直接。

风险：

- 环境可复现性最差；
- 容易出现“安装成功但某些场景延迟爆炸”的隐患；
- 不适合写进正式依赖策略。

## 推荐设计

推荐按以下顺序决策：

1. **先验证 `omegaconf` 更新版本是否放宽了 `antlr4` 约束**；
2. **若放宽，则采用方案 A**，在 `pyproject.toml` 中显式提升 `omegaconf` 版本下界；
3. **若未放宽，则退回方案 B**，让默认依赖保留基础 `lm_eval`，但把 `math` 相关链路移到 optional extra；
4. **不采用方案 C 作为正式项目策略**，最多只作为本地临时环境应急办法。

## 文件范围

本次设计的主要落点文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml`

按需同步文档的候选文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/README.md`

明确不属于本设计范围的文件：

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/dllm/pipelines/oneflow/sampler.py`

## 实施原则

- 先调查上游依赖约束，再修改 `pyproject.toml`；
- 依赖策略应反映“新环境如何正确安装”，而不是反映“当前容器里碰巧能装成什么样”；
- 如果最终采用 optional extra 路线，README 必须给出清晰安装说明；
- 若 `omegaconf` 升级后与项目代码不兼容，应立即回退到依赖分层方案，而不是继续堆补丁。

## 验收标准

### 若采用方案 A（升级 `omegaconf`）

- `/mnt/ai4s/zhangjinouwen/Project/dllm/oneflow/dllm/pyproject.toml` 明确允许一个兼容 `antlr4 4.11` 的 `omegaconf` 版本区间；
- 安装 `dllm` 与 `lm_eval` 时不再出现 `omegaconf 2.3.0` 风格的 resolver 冲突；
- 现有代码对 `omegaconf` 的基本使用不受影响。

### 若采用方案 B（拆分 `math` extra）

- 默认 `pip install -e .` 不再触发 `antlr4` 版本冲突；
- 需要数学评测的人可通过额外安装路径获得对应能力；
- README 对安装方式的描述与 `pyproject.toml` 一致。

## 预期结果

完成这轮依赖对齐后，项目应具备以下特征：

- `lm_eval` 的地位在依赖图中清晰、稳定；
- `omegaconf` 不再因为历史解析结果而被误认为必须锁在 `2.3.0`；
- 新环境安装不再依赖“先装这个再装那个”的手工顺序技巧；
- 后续如需提交 `sampler.py` 的兼容性修补，可以独立进行，不与本次依赖决策互相污染。
