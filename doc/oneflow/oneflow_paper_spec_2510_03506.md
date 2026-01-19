# OneFlow 论文可核对规格（arXiv:2510.03506）

目的：把论文中**明确写出来**、可直接用来做代码审计的“算法/公式/训练目标”摘出来，作为本仓库 OneFlow 实现的对齐基准。

论文来源：[arXiv:2510.03506](https://arxiv.org/html/2510.03506)

---

## 1. 文本：Edit Flows 插入建模（Section 2.1）

### 1.1 模型输出与采样率（Eq. 3）

论文给出的插入 CTMC 一步转移（忽略 \(o(h)\)）：

- 模型在每个位置 \(i\) 输出：
  - \(\lambda^{i}(X_t)\in \\mathbb{R}^+\)：预测“在 \(i,i+1\) 之间缺失 token 的数量”
  - \(Q^{i}(a\\mid X_t)\)：预测缺失 token 的分布（bag-of-tokens）
- 采样时（Eq. 3）：
  \[
  \\mathbb{P}(X_{t+h}=\\text{ins}(X_t,i,a)\\mid X_t)=h\\frac{\\dot\\kappa_t}{1-\\kappa_t}\\lambda^{i}(X_t)Q^{i}(a\\mid X_t)
  \]

重要实现说明（论文原文）：作者将 \(\\frac{\\dot\\kappa_t}{1-\\kappa_t}\) 从 rate 中“因子化”出来，并采用 **t-independent** 的插入预测（“实践中不把 time 喂给网络来预测 insertions”）。

### 1.2 Zero-inflated missing-count：\(\pi\) 与 \(\lambda_{\\text{nonzero}}\)（Eq. 5）

论文指出缺失计数 \(k^i\) 在 \(k^i=0\) 处质量很大，因此显式建模“零插入”概率：

- \(\mathbb{P}(k=0)=\\pi\)
- \(k>0\) 时：\(\mathbb{P}(k)=(1-\\pi)\\,\\text{Pois}(k;\\lambda_{\\text{nonzero}}\\mid k>0)\)

训练方式（论文原文）：
- 用 **BCE** 训练 \(\pi\)（判断 \(k^i\) 是否为 0）
- 对 **nonzero counts** 用原始 Poisson loss（Eq. 4）训练 \(\lambda_{\\text{nonzero}}\)
- 采样时建议：先用 \(\pi\) 采样“是否 0 插入”，若非 0，再用 \(\lambda_{\\text{nonzero}}\) 控制插入

### 1.3 Bag-of-tokens loss（Eq. 6）

对每个位置 \(i\)，令 \(\mathcal{A}_i\) 为 \(X_t^i\) 与 \(X_t^{i+1}\) 间被删掉的 token 多重集合，则：
\[
\\ell_{\\text{tokens}}(Q^i)=-\\sum_{a\\in \\mathcal{A}_i}\\log Q^i(a\\mid X_t)
\]

### 1.4 文本 combined loss（Eq. 7）与“不要乘 \(\\dot\\kappa/(1-\\kappa)\)”

论文给出的文本总 loss（Eq. 7，原式）：
\[
\\mathcal{L}_{\\text{text}}=
\\mathbb{E}_{t, X_t\\mid X_1}\\Big[
\\frac{1}{n}\\sum_{i=1}^{n}\\ell_{\\text{tokens}}(Q^i)
+\\ell_{\\text{Poisson}}(\\lambda^{i}_{\\text{nonzero}})\\,\\mathbf{1}_{[k_i>0]}
+\\ell_{\\text{BCE}}(\\pi^{i})
\\Big]
\]

并明确说明：这与 Edit Flows 原目标的区别之一是**不再用** \(\\frac{\\dot\\kappa_t}{1-\\kappa_t}\) 对 loss 加权；论文称该加权“不影响最优解”，但他们经验上“不用效果更好”。

---

## 2. 图像：Flow Matching（Section 2.2）

训练采用线性插值噪声（Eq. 9）：
- \(Y_0\\sim\\mathcal{N}(0,I)\)
- \(Y_t=tY_1+(1-t)Y_0\)
- 监督 velocity：\(\|v(Y_t,t)-(Y_1-Y_0)\|^2\)

---

## 3. 交错时间表：Interleaved schedule（Section 2.3.2 + Eq. 28/29 + Appendix E）

论文对 interleaved schedule 的关键点（Section 2.3.2）：
- 采样 extended time：\(\tau_{\\text{text}}\\in[0,2]\)
  - token 是否出现在 \(X_t\) 的概率由 \(\kappa(\\min\\{1,\\tau_{\\text{text}}\\})\) 决定
  - 直觉：`<|image|>` 最晚在 \(\tau=1\) 插入，图像最晚到 \(\tau=2\) fully denoise
- 每张图像：
  - \(u\\sim\\text{Unif}(0,1)\)
  - \(\tau_{\\text{img}}=\\tau_{\\text{text}}-\\kappa^{-1}(u)\)（Eq. 28）
  - 若 \(\tau_{\\text{img}}<0\)：该 snapshot “图像尚未插入”，等价于 `<|image|>` 作为**被删 token**进入某个 \(\mathcal{A}_i\)（从而由文本插入 loss 学会插入图像 token）
  - 否则设 \(t_{\\text{img}}=\\min\\{1,\\tau_{\\text{img}}\\}\)（Eq. 29）并训练图像 flow matching

Appendix E 的 Algorithm 2 也明确：\(\pi\) gate 是可选项（若不用 \(\pi\) 的 parameterization，则跳过该步）。

---

## 4. κ 调度（Appendix B.6）

论文在 B.6 讨论 \(\\kappa_t=t^k\) 的 k-scheduler（线性/二次/三次），并报告：**线性（k=1）最好**，更高次会导致过激的 token deletion。


