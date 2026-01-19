from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
from dllm.core.schedulers import BaseKappaScheduler, LinearKappaScheduler
from dllm.pipelines.ctmc_utils import sample_from_logits
from dllm.pipelines.oneflow.sequence_ops import build_unified_sampler_inputs_bs1
from dllm.pipelines.oneflow.sampler_ops import apply_insertions_right_to_left, p_lam, p_pi
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_TOKEN


@dataclass
class OneFlowSamplerConfig(SamplerConfig):
    dt: float = 0.05
    time_epsilon: float = 1e-3
    max_steps: int = 512
    temperature: float = 0.0
    use_pi_gate: bool = True
    # Safety/perf guards for sampling (to avoid attention OOM when the sequence explodes).
    # - `max_new_tokens`: cap growth relative to the initial prompt length.
    # - `max_seq_len`: cap the unified sequence length (text + modality tokens).
    # - `max_insertions_per_step`: cap parallel insertions per step (keeps growth ~linear).
    max_new_tokens: int | None = None
    max_seq_len: int | None = None
    max_insertions_per_step: int | None = None
    # If True, only allow insertions at the END of the current sequence (append-only).
    # This behaves closer to autoregressive completion and avoids \"filling\" earlier slots.
    append_only: bool = False
    # Optional: suppress very common whitespace-only tokens (e.g., GPT2 token id for \" \")
    # during sampling to avoid degenerate space loops (useful for language sanity checks).
    suppress_whitespace_tokens: bool = False
    suppress_token_ids: list[int] | None = None
    # Clamp scheduler weight w(t)=κ'(t)/(1-κ(t)) during sampling to avoid blow-up near t→1.
    # (For Linear κ(t)=t, w(t)=1/(1-t) diverges.)
    max_w: float | None = None
    # Paper (arXiv:2510.03506, Sec 2.1.1): insertion predictions are t-independent in practice.
    # If False, we feed a constant time value for *text tokens* (π/λ/Q do not depend on t_text).
    # Image latent tokens still use their own t_img.
    condition_text_on_time: bool = False
    image_num_tokens: int = 64  # fixed number of latent tokens per image (v1)
    edit_prompt: bool = False  # if False, only allow insertions after the prompt


@dataclass
class OneFlowSamplerOutput(SamplerOutput):
    images: list[torch.Tensor] | None = None
    image_times: list[float] | None = None


class OneFlowSampler(BaseSampler):
    """
    Interleaved text-image sampler (Algorithm 1-2).

    v1 will support bs=1 initially (same as EditFlowSampler), then generalize to bs>1.
    """

    kappa_scheduler: BaseKappaScheduler | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.kappa_scheduler is None:
            self.kappa_scheduler = LinearKappaScheduler()

    @torch.no_grad()
    def sample(
        self,
        inputs: List[torch.Tensor | list],
        config: Optional[OneFlowSamplerConfig] = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = OneFlowSamplerConfig()

        dt = float(kwargs.get("dt", config.dt))
        time_epsilon = float(kwargs.get("time_epsilon", config.time_epsilon))
        max_steps = int(kwargs.get("max_steps", config.max_steps))
        temperature = float(kwargs.get("temperature", config.temperature))
        use_pi_gate = bool(kwargs.get("use_pi_gate", config.use_pi_gate))
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_seq_len = kwargs.get("max_seq_len", config.max_seq_len)
        max_insertions_per_step = kwargs.get(
            "max_insertions_per_step", config.max_insertions_per_step
        )
        append_only = bool(kwargs.get("append_only", config.append_only))
        suppress_whitespace_tokens = bool(
            kwargs.get("suppress_whitespace_tokens", config.suppress_whitespace_tokens)
        )
        suppress_token_ids = kwargs.get("suppress_token_ids", config.suppress_token_ids)
        max_w = kwargs.get("max_w", config.max_w)
        condition_text_on_time = bool(
            kwargs.get("condition_text_on_time", config.condition_text_on_time)
        )
        image_num_tokens = int(kwargs.get("image_num_tokens", config.image_num_tokens))
        edit_prompt = bool(kwargs.get("edit_prompt", config.edit_prompt))
        return_dict = bool(kwargs.get("return_dict", config.return_dict))

        if len(inputs) != 1:
            raise NotImplementedError("OneFlowSampler v1 only supports bs=1")

        x0 = inputs[0]
        if isinstance(x0, list):
            x_list = [int(t) for t in x0]
        else:
            x0 = x0.detach().to("cpu")
            if x0.dim() == 2:
                if x0.size(0) != 1:
                    raise NotImplementedError("OneFlowSampler v1 only supports bs=1")
                x0 = x0.squeeze(0)
            x_list = [int(t) for t in x0.tolist()]

        bos = self.tokenizer.bos_token_id
        if bos is None:
            raise ValueError("tokenizer.bos_token_id must be set")
        if len(x_list) == 0:
            x_list = [int(bos)]
        elif x_list[0] != int(bos):
            x_list = [int(bos)] + x_list

        prompt_len = len(x_list)

        image_token_id = int(self.tokenizer.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))
        if self.tokenizer.unk_token_id is not None and image_token_id == int(
            self.tokenizer.unk_token_id
        ):
            raise ValueError(
                f"Tokenizer does not recognize {ONEFLOW_IMAGE_TOKEN}. "
                "Please add it as a special token before sampling."
            )

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        device = next(self.model.parameters()).device

        dim_latent = int(getattr(getattr(self.model, "config", None), "dim_latent", 4))
        zero_lat = torch.zeros((dim_latent,), device=device, dtype=torch.float32)

        # images are stored aligned to the order of `<|oneflow_image|>` tokens in x_list
        images: list[dict[str, Any]] = []

        def ensure_images_for_prompt():
            nonlocal images
            n_tokens = sum(1 for t in x_list if int(t) == int(image_token_id))
            while len(images) < n_tokens:
                images.append(
                    dict(
                        latent=torch.randn(
                            (image_num_tokens, dim_latent),
                            device=device,
                            dtype=torch.float32,
                        ),
                        t=0.0,
                    )
                )

        ensure_images_for_prompt()

        # Build token suppression list once (if requested).
        suppress: set[int] = set()
        if suppress_token_ids:
            try:
                suppress.update(int(x) for x in suppress_token_ids)
            except Exception:
                pass
        if suppress_whitespace_tokens:
            # Common whitespace artifacts for GPT-like tokenizers.
            for s in [" ", "\xa0", "\t"]:
                try:
                    ids = self.tokenizer.encode(s, add_special_tokens=False)
                    if isinstance(ids, list) and len(ids) == 1:
                        suppress.add(int(ids[0]))
                except Exception:
                    continue
        suppress_list = sorted(suppress) if suppress else None

        # histories (text-only for v1)
        histories = [] if return_dict else None

        t_text = 0.0
        for _step in range(max_steps):
            # Hard cap on sequence growth (pre-forward) to avoid attention OOM.
            if max_new_tokens is not None:
                try:
                    if (len(x_list) - int(prompt_len)) >= int(max_new_tokens):
                        break
                except Exception:
                    pass

            ensure_images_for_prompt()

            # ---- build unified sequence (text + modality tokens) ------------------
            uni = build_unified_sampler_inputs_bs1(
                x_list=x_list,
                images=images,
                t_text=float(t_text),
                image_token_id=int(image_token_id),
                pad_id=int(pad_id),
                dim_latent=int(dim_latent),
                condition_text_on_time=bool(condition_text_on_time),
                device=device,
            )

            input_ids = uni.input_ids
            attention_mask = uni.attention_mask
            is_any_modality = uni.is_any_modality
            modality_tokens = uni.modality_tokens
            times_tensor = uni.times
            mod_pos = uni.modality_positions
            text_pos_total = uni.text_pos_total
            image_slices = uni.image_slices

            if max_seq_len is not None:
                try:
                    if int(input_ids.shape[1]) > int(max_seq_len):
                        break
                except Exception:
                    pass

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_any_modality=is_any_modality,
                modality_tokens=modality_tokens,
                modality_positions=mod_pos,
                times=times_tensor,
            )

            pi = out["pi"][0]  # [N]
            lam = out["lambda_nonzero"][0]  # [N]
            q_logits = out["q_logits"][0]  # [N,V]
            v = out["v"][0]  # [N,dim_latent]

            # ---- image updates (Euler) -------------------------------------------
            for img_idx, (start, end) in enumerate(image_slices):
                t_img = float(images[img_idx]["t"])
                if t_img >= 1.0 - time_epsilon:
                    continue
                dt_img = min(dt, 1.0 - t_img)
                images[img_idx]["latent"] = images[img_idx]["latent"] + dt_img * v[
                    start:end
                ]
                images[img_idx]["t"] = t_img + dt_img

            # ---- text insertions (parallel) --------------------------------------
            dt_text = min(dt, 1.0 - t_text)
            if dt_text > 0.0:
                t_tensor = torch.tensor([[t_text]], device=device, dtype=torch.float32)
                w = float(self.kappa_scheduler.weight(t_tensor).item())
                if max_w is not None:
                    try:
                        w = min(float(w), float(max_w))
                    except Exception:
                        pass

                insertions: list[tuple[int, int]] = []  # (slot_index_in_x, token_id)
                for i, pos in enumerate(text_pos_total):
                    if append_only and i != (len(x_list) - 1):
                        continue
                    # By default, do not edit the prompt: only allow insertions after the
                    # last prompt token. This keeps prefix conditioning stable.
                    if (not edit_prompt) and i < (prompt_len - 1):
                        continue
                    prob_lam = p_lam(dt_text=dt_text, w=w, lam_nonzero=float(lam[pos].item()))
                    do_lam = bool(torch.bernoulli(torch.tensor(prob_lam, device=device)).item())
                    if not do_lam:
                        continue

                    if use_pi_gate:
                        prob_pi = p_pi(pi=float(pi[pos].item()))
                        do_pi = bool(torch.bernoulli(torch.tensor(prob_pi, device=device)).item())
                        if not do_pi:
                            continue

                    a = sample_from_logits(
                        q_logits[pos],
                        temperature=float(temperature),
                        suppress_token_ids=suppress_list,
                    )
                    insertions.append((i, a))

                # ---- cap insertions to avoid runaway growth ----------------------
                cap: int | None = None
                if max_insertions_per_step is not None:
                    try:
                        cap = int(max_insertions_per_step)
                    except Exception:
                        cap = None
                if max_new_tokens is not None:
                    try:
                        remaining = int(max_new_tokens) - (len(x_list) - int(prompt_len))
                        remaining = max(0, remaining)
                        cap = remaining if cap is None else min(cap, remaining)
                    except Exception:
                        pass
                if cap is not None and cap >= 0 and len(insertions) > cap:
                    # Prefer keeping insertions closer to the end (better for prompt completion).
                    insertions = sorted(insertions, key=lambda x: x[0], reverse=True)[:cap]

                # apply from right to left so indices stay valid
                apply_insertions_right_to_left(
                    x_list=x_list,
                    insertions=insertions,
                    image_token_id=int(image_token_id),
                    images=images,
                    make_new_image=lambda: dict(
                        latent=torch.randn(
                            (image_num_tokens, dim_latent),
                            device=device,
                            dtype=torch.float32,
                        ),
                        t=0.0,
                    ),
                )

                t_text = min(1.0, t_text + dt_text)

            if histories is not None:
                histories.append(torch.tensor([x_list], device=device, dtype=torch.long))

            # stop if both text and all images are done
            if t_text >= 1.0 - time_epsilon and all(
                float(img["t"]) >= 1.0 - time_epsilon for img in images
            ):
                break

        seq = torch.tensor([x_list], device=device, dtype=torch.long)
        if not return_dict:
            return seq

        return OneFlowSamplerOutput(
            sequences=seq,
            histories=histories,
            images=[img["latent"] for img in images] if images else None,
            image_times=[float(img["t"]) for img in images] if images else None,
        )

    @torch.no_grad()
    def infill(
        self,
        inputs: List[torch.Tensor | list],
        config: SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError("OneFlowSampler.infill is not implemented in v1.")


