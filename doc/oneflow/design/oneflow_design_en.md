# OneFlow Multimodal Design (Concise, EN)

This is a concise English summary. For the full Chinese design with implementation details, see:
- `doc/oneflow/design/oneflow_design_zh.md`
- Paper specs: `doc/oneflow/design/oneflow_paper_spec_2510_03506.md`
- Code alignment audit: `doc/oneflow/validation/oneflow_paper_alignment_audit_2510_03506.md`

---

## 1. Goal & Scope
- Add OneFlow pipeline (`dllm/pipelines/oneflow/`) with training + sampling.
- Text: insertion-based Edit Flow; Image: latent flow matching.
- Interleaved schedule κ to co-generate text and images.
- Emphasis: algorithm correctness and extensible engineering; not full paper-scale recipe.

---

## 2. Core Architecture
- **Unified trunk** (Transfusion-style Transformer) for mixed tokens.
- **Text heads**: π, λ_nonzero, Q for insertion modeling.
- **Image head**: velocity v(Y_t, t) for flow matching.

Key refs:
- Trunk & modality mask logic from `reference/transfusion-pytorch` (vendored with license).
- Text insertion modeling per paper Eq(3–7).

---

## 3. Training (Algorithm 3)
- Sample `τ_text` in [0,1] (or [1,2] with `mixed_generation_prob`), set `t_text=min(1, τ_text)`.
- Keep tokens with prob κ(t_text); build `X_t` + bag-of-tokens `A_i`.
- **Text loss**: Eq(7) = token CE + π BCE + Poisson(λ_nonzero, k>0), **no** `w(t)` weighting.
- **Image loss**: interleaved schedule `τ_img = τ_text - κ^{-1}(u)` and flow matching `||v(Y_t,t)-(Y1-Y0)||^2`.
- Default alignment: `text_loss_type="paper"`, `condition_text_on_time=False`.

---

## 4. Inference (Algorithm 1–2)
- Maintain `X` (text), `I` (image latents with t_img), `t_text`.
- Each step: update images by Euler; update text by parallel insertions with `p_i^λ` and optional `p_i^π`.
- Inserted `<|oneflow_image|>` spawns a new latent with `t_img=0`.

---

## 5. Data & Minimal Format
- Minimal sample: `(caption, image)`.
- Tokenized text: `[BOS] + tokens + [<|oneflow_image|>] + [EOS]`.
- Image stored as latent `Y1` (online or offline).

---

## 6. Repo Integration
- Paths: `dllm/pipelines/oneflow/`, `examples/oneflow/`.
- Engineering refactor (CTMC helpers + `prompt_len` semantics) is documented separately:
  - `doc/oneflow/engineering/oneflow_editflow_utils_reuse_plan_zh.md`

---

## 7. Milestones
- Text-only toy run (variable-length insertion works).
- Text+image toy run (sample at least one image latent, optional VAE decode).
- Clean, extensible structure for future extensions.

