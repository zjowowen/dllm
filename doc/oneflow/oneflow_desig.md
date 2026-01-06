# OneFlow Multimodal (Text Edit Flow + Image Flow Matching) Design & Implementation Plan

> Goal: integrate OneFlow (insertion-based text edit flow + image-latent flow matching + interleaved κ schedule) into this repo, following the existing `dllm/pipelines/editflow` engineering style.
>
> References:
> - OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows — [arXiv:2510.03506](https://arxiv.org/html/2510.03506)
> - Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model — `https://arxiv.org/html/2408.11039`
> - Edit Flows (prior work; for CTMC loss style & repo alignment): `https://arxiv.org/html/2506.09018`

---

## Overview

- This repo focuses on diffusion-style language models and provides training/inference scaffolding; `editflow` already implements CTMC-style edit operations.
- OneFlow adds **multimodality**: insertion-based edit flow for text + flow matching in image latent space, tied together with an **interleaved κ schedule** for concurrent generation.
- You provided `reference/transfusion-pytorch`; we use it as the reference backbone for image-latent modeling and the unified trunk (vendored / internalized under `dllm`).

---

## 1. Goals & Non-goals

### Goals
- Add `dllm/pipelines/oneflow/` with model, trainer, sampler, collator, utilities.
- Implement OneFlow Algorithm 3 (interleaved training):
  - Text: κ-keep deletion noise → bag-of-tokens targets → insertion intensities λ and token distribution Q.
  - Images: use `τ_img = τ_text - κ^{-1}(u)` to decide “inserted yet”; flow-match the inserted latents.
- Implement OneFlow Algorithm 1-2 (interleaved inference):
  - Text: parallel insertions (Bernoulli triggers).
  - Images: Euler step on latents (ODE solver optional).
- Provide runnable scripts under `examples/oneflow/`.
- Document data format, training stages, hyperparameters, dependencies, and license notes.

### Non-goals (v1)
- No attempt to exactly reproduce the full paper recipe (data mixtures, scale, exact metrics). The focus is correctness + extensibility.
- No hard requirement on a specific large pretrained LLM init; v1 targets a from-scratch unified trunk + heads, with room for later upgrades.

---

## 2. Data & Representation

## 2.1 Ground-truth sample structure

OneFlow training data is an interleaved sequence containing:
- **Text tokens**: discrete sequence \(X\)
- **Image latents**: continuous tensors \(Y_1\) (one latent tensor per image)
- **Image positions in sequence**: we must bind each image to a specific insertion slot (gap), so that when `τ_img<0` we can put `<|oneflow_image|>` back into the correct bag `𝒜_j`.

Recommended internal representation:
- Each sample is a Python list of segments:
  - Text segment: `torch.LongTensor[seq]`
  - Image segment: `torch.FloatTensor[...]` (latent)
- Use a special token `<|oneflow_image|>` inside the text as an image placeholder, and place the corresponding latent segment right after it.

## 2.2 Collator outputs

`OneFlowCollator` collates a batch into trainer-ready structures, at minimum:
- `x1_ids`: ground-truth token ids (incl. BOS/EOS and `<|oneflow_image|>`)
- `image_latents` (optional): pre-encoded latents aligned to the `<|oneflow_image|>` tokens
- optional `prompt_len`: for prompt masking in SFT

---

## 3. Model Architecture

## 3.1 High-level

`OneFlowModel` (`dllm/pipelines/oneflow/models/oneflow_model.py`) has three parts:
- **Unified trunk (Transfusion-style Transformer)** for mixed tokens (text embeddings + image latent projections), with per-token time conditioning.
- **Text insertion heads** producing `{π, λ_nonzero, Q}`.
- **Image flow head** producing per-latent-token velocity \(v(Y_t, t)\) (or equivalent flow).

## 3.2 Unified trunk (Transfusion backbone)

We vendor the trunk-related modules from `reference/transfusion-pytorch` into `dllm/third_party/transfusion_pytorch` (with MIT license attribution), mainly leveraging:
- time conditioning (Random Fourier + AdaLN-like conditioning)
- rotary embedding with relative positions derived from modality positions
- mixed attention masks driven by `modality_positions`

Note: Transfusion also supports AR next-token loss for text. OneFlow text is non-AR; we reuse the trunk/utilities and implement OneFlow insertion heads/losses on top.

## 3.3 Text insertion heads

- Insertion slots: for each token position \(i\) in current text \(X_t\) (including BOS, typically excluding EOS), define the slot “insert after i”.
- Head outputs:
  - `pi = sigmoid(W_pi h_i)` → \(π^i\)
  - `lambda_nonzero = softplus(W_lam h_i)` → \(λ_{\text{nonzero}}^i\)
  - `Q_logits = W_q h_i` → softmax gives \(Q^i(\cdot)\)
- `Q` vocabulary must include normal text tokens and `<|oneflow_image|>` (spawns an image latent).

## 3.4 Image head (flow / velocity)

For image-latent token positions, predict a latent-dim vector:
- `v = W_img h` (linear or small MLP)
for flow matching with target \(Y_1 - Y_0\).

---

## 4. Training Plan (Algorithm 3)

## 4.1 κ schedule and inverse

We reuse `dllm/core/schedulers/kappa.py`:
- `kappa(t)`, `kappa_derivative(t)`, `weight(t)=κ'(t)/(1-κ(t))`

OneFlow needs `κ^{-1}(u)` (paper Eq. 28). Implementation strategy:
- Linear: \(κ^{-1}(u)=u\)
- Cosine: \(κ^{-1}(u)=\\frac{2}{\\pi}\\arccos(1-u)\)
- Others: numeric bisection fallback on `[0,1]`

## 4.2 Text noising and bag-of-tokens

Given ground-truth \(X\):
- Sample `τ_text ~ Unif[0,2]`, set `t_text = min(1, τ_text)`.
- κ-keep each token with prob `κ(t_text)`; always keep BOS.
- Build:
  - `X_t`: subsequence of kept tokens
  - `A_j`: per-slot bag-of-tokens for missing tokens

## 4.3 Text loss (insertion edit flow)

At `t_text`, `w = κ'(t)/(1-κ(t))`. The model predicts `λ_i` and `Q_i` per slot.

CTMC NLL MC estimator (aligned with `editflow` survival + positive terms):
- **Survival**:
  - \(L_{surv} = E[w \\sum_i λ_i]\)
- **Positive**:
  - \(L_{pos} = E[w \\sum_i \\sum_{a\\in A_i}(-\\log λ_i - \\log Q_i(a))]\)
- Optional π gating term (BCE), or use π only at sampling time.

## 4.4 Image interleaved times & flow matching

For each image with ground-truth latent `Y1`:
- sample `u~Unif(0,1)`, compute `τ_img = τ_text - κ^{-1}(u)`
- if `τ_img < 0`: image is “not inserted yet” at this snapshot → add `<|oneflow_image|>` into the proper bag `A_j`
- else:
  - `t_img = clip(τ_img)` into `[0,1]`
  - sample `Y0 ~ N(0,I)`
  - `Y_t = t_img*Y1 + (1-t_img)*Y0`
  - target `flow = Y1 - Y0`
  - `L_image = MSE(vθ(Y_t, t_img), flow)`

## 4.5 Total loss

- `L = L_text + w_img * L_image (+ optional terms)`
- Default: `w_img=1.0` and normalize image loss by number of latent tokens.

## 4.6 Recommended stages

- Stage A: text-only insertion flow
- Stage B: image-only latent flow matching (caption-conditioned or unconditional)
- Stage C: mixed-modal interleaved training
- Stage D: multimodal instruction tuning (VQA/dialog)

---

## 5. Inference (Algorithm 1-2)

## 5.1 State variables

Sampler maintains:
- `X`: current text token sequence (dynamic length)
- `I`: set/list of image latents, each with `t_img`
- `t_text`: text time (0→1)

## 5.2 One step

Each step `Δt`:
1) forward `model(X,I,t)` → `{π,λ,Q}` and image velocities `v`
2) **images**: Euler update for all `t_img<1`
3) **text**: parallel insertions using `p_i^λ` and optional `p_i^π`; insert token `a~Q_i`; if `a` is `<|oneflow_image|>`, spawn a new latent `Y~N(0,I)`
4) update `t_text`

Stop when `t_text>=1` and all images reach `t_img>=1`.

## 5.3 Decoding

If VAE is used:
- Decode each final latent `Y` with `VAEDec`.
Otherwise:
- Return latents directly.

---

## 6. Datasets

- Text PT: FineWeb / C4 / The Pile / OpenWebText (toy: tiny-shakespeare)
- Image-text PT: LAION subsets, CC3M/CC12M, COCO Captions, Flickr30k, DataComp
- Multimodal instruction: VQAv2, OK-VQA, GQA, LLaVA-style, ShareGPT4V

Minimal schema (recommended v1):
- `(caption, image)` → tokens `[BOS] + tokenize(caption) + [<|oneflow_image|>] + [EOS]` + image latent `Y1`

---

## 7. Repo Integration & APIs

New paths:
- `dllm/pipelines/oneflow/`
- `examples/oneflow/`
- `doc/oneflow/oneflow_desig.md` (this file)

Align with existing repo scripts:
- training scripts mirror `examples/editflow/pt.py` / `sft.py`
- inference scripts mirror `examples/editflow/sample.py` / `chat.py`

---

## 8. Dependencies & License

- Transfusion backbone deps are added as `oneflow` optional extras in this repo (to avoid impacting default installs).
- Vendored code preserves upstream MIT license and attribution under `dllm/third_party/transfusion_pytorch/`.

---

## 9. Milestones & Acceptance

- Text-only toy run works: variable-length insertion sampling grows sequences.
- Text+image toy run works: can sample at least one image latent and optionally decode with VAE.
- Clean, extensible structure for future upgrades (pretrained init, multi-image, multi-modality, variable resolution).


