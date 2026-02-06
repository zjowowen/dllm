from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from dllm.core.schedulers import BaseKappaScheduler
from dllm.pipelines.ctmc_utils import pad_1d


@dataclass
class NoisedBagsOutput:
    """
    Output of the discrete noising step (Algorithm 3, lines 5-14).

    - xt_list[b] is the kept-token sequence X_t for sample b (python list of ints)
    - bags_list[b][i] is the bag-of-tokens A_i aligned to xt_list[b][i]
    - keep_mask_list[b] is a per-token keep mask over x1_ids[b] (for tracing/debug)
    """

    xt_list: list[list[int]]
    bags_list: list[list[list[int]]]
    keep_mask_list: list[list[bool]]


@dataclass
class InterleavedImagesOutput:
    """
    Output of interleaved image schedule (Algorithm 3, lines 16-28) for a batch.

    For each sample b:
    - kept_images_list[b] aligns to the remaining `<|image|>` tokens in xt_list[b]
    - kept_timg_list[b] contains per-image t_img values (same order as kept_images_list[b])
    """

    xt_list: list[list[int]]
    bags_list: list[list[list[int]]]
    kept_images_list: list[list[torch.Tensor]]
    kept_timg_list: list[list[float]]


@dataclass
class UnifiedTrainBatch:
    """
    Unified sequence tensors for OneFlowModel forward.

    Shapes:
    - input_ids:        [B, L]
    - attention_mask:   [B, L]
    - is_any_modality:  [B, L] (bool)
    - modality_tokens:  [B, L, dim_latent] (float32)
    - flow_targets:     [B, L, dim_latent] (float32) (zeros on text positions)
    - times:            [B, L] (float32)
    - modality_positions: [B, M, 3] long or None

    Also returns `xt_to_total_pos_list[b]`: mapping from X_t token index -> unified position.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    is_any_modality: torch.Tensor
    modality_tokens: torch.Tensor
    flow_targets: torch.Tensor
    times: torch.Tensor
    modality_positions: torch.Tensor | None
    xt_to_total_pos_list: list[list[int]]


@dataclass
class UnifiedSampleInputs:
    """
    Unified sequence inputs for OneFlowSampler (bs=1).

    Shapes:
    - input_ids:        [1, L]
    - attention_mask:   [1, L]
    - is_any_modality:  [1, L] bool
    - modality_tokens:  [1, L, dim_latent]
    - times:            [1, L]
    - modality_positions: [1, M, 3] long or None

    Extra mappings:
    - text_pos_total[i] is the unified position for x_list[i]
    - image_slices[j] is (start,end) range of modality tokens for image j in unified positions
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    is_any_modality: torch.Tensor
    modality_tokens: torch.Tensor
    times: torch.Tensor
    modality_positions: torch.Tensor | None
    text_pos_total: list[int]
    image_slices: list[tuple[int, int]]


def sample_tau_text(
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample τ_text ~ Unif[0, 2] (Algorithm 3, line 2).

    When τ_text > 1, t_text = min(1, τ_text) = 1, meaning text is fully
    unmasked while images may still be noised — the “mixed generation” regime
    described in the paper (Sec 3.0.1).
    """
    B = int(batch_size)
    return torch.rand((B, 1), device=device) * 2.0


def tau_to_t_text(tau_text: torch.Tensor) -> torch.Tensor:
    """t_text = min(1, τ_text)."""
    return torch.minimum(tau_text, torch.ones_like(tau_text))


def flatten_latent_to_tokens(y: torch.Tensor, *, dim_latent: int) -> torch.Tensor:
    """
    Standardize a latent tensor into a token sequence [N, dim_latent].

    Accepts:
      - [N, d]
      - [d, H, W] (channel-first)
      - [H, W, d] (channel-last)
    """
    d = int(dim_latent)
    if y.dim() == 2 and int(y.shape[-1]) == d:
        return y
    if y.dim() == 3:
        if int(y.shape[0]) == d:
            y = y.permute(1, 2, 0).contiguous()  # [H,W,d]
        elif int(y.shape[-1]) != d:
            raise ValueError(f"Unrecognized latent shape: {tuple(y.shape)} (expected dim_latent={d})")
        return y.reshape(-1, d)
    raise ValueError(f"Unrecognized latent tensor ndim={y.dim()} shape={tuple(y.shape)}")


def build_noised_xt_and_bags(
    *,
    x1_ids: list[list[int]],
    kappa_keep: torch.Tensor,
    device: torch.device,
    prompt_len_list: list[int] | None = None,
    image_token_id: int | None = None,
    disallow_image_in_prompt: bool = True,
) -> NoisedBagsOutput:
    """
    Build X_t and bag-of-tokens A_i by κ-keep deletion.

    - Always keeps BOS (x1_ids[b][0]).
    - If prompt_len_list is provided, always keeps tokens < prompt_len.
    - If image_token_id is provided, always keeps `<|image|>` tokens (so image deletion
      is governed by the interleaved τ_img schedule, not κ-keep).
    """
    if kappa_keep.dim() == 2 and kappa_keep.shape[1] == 1:
        k_list = kappa_keep.squeeze(1).tolist()
    else:
        k_list = kappa_keep.reshape(-1).tolist()

    if len(k_list) != len(x1_ids):
        raise ValueError(f"kappa_keep batch mismatch: got {len(k_list)} values, expected {len(x1_ids)}.")

    xt_list: list[list[int]] = []
    bags_list: list[list[list[int]]] = []
    keep_mask_list: list[list[bool]] = []

    for b_idx, (x1, kb) in enumerate(zip(x1_ids, k_list)):
        if not x1:
            raise ValueError("Empty x1_ids is not supported for OneFlow training.")

        pl: int | None = None
        if prompt_len_list is not None:
            pl = int(prompt_len_list[b_idx])
            if pl <= 0 or pl > len(x1):
                raise ValueError(f"Invalid prompt_len={pl} for sample {b_idx}: x1 length={len(x1)}")
            if disallow_image_in_prompt and image_token_id is not None:
                if any(int(t) == int(image_token_id) for t in x1[:pl]):
                    raise ValueError(
                        "prompt_len spans over image token; conditioning on prompt images is not supported."
                    )

        keep = (torch.rand(len(x1), device=device) < float(kb)).tolist()
        keep[0] = True  # BOS
        if pl is not None:
            for i in range(pl):
                keep[i] = True
        if image_token_id is not None:
            for i, tok_id in enumerate(x1):
                if int(tok_id) == int(image_token_id):
                    keep[i] = True

        xt: list[int] = []
        bags: list[list[int]] = []
        for tok_id, is_keep in zip(x1, keep):
            if is_keep:
                xt.append(int(tok_id))
                bags.append([])
            else:
                bags[-1].append(int(tok_id))

        xt_list.append(xt)
        bags_list.append(bags)
        keep_mask_list.append(keep)

    return NoisedBagsOutput(xt_list=xt_list, bags_list=bags_list, keep_mask_list=keep_mask_list)


def apply_interleaved_image_schedule(
    *,
    x1_ids: list[list[int]],
    xt_list: list[list[int]],
    bags_list: list[list[list[int]]],
    image_latents_raw: list[Any],
    tau_text: torch.Tensor,
    scheduler: BaseKappaScheduler,
    image_token_id: int,
    device: torch.device,
) -> InterleavedImagesOutput:
    """
    Apply interleaved image schedule per sample (Algorithm 3, lines 16-28).

    This mutates X_t by deleting some `<|image|>` tokens when τ_img < 0, and merges bags so
    `bags` remains aligned with the remaining X_t token positions.
    """
    B = len(x1_ids)
    if len(xt_list) != B or len(bags_list) != B:
        raise ValueError("Batch size mismatch in apply_interleaved_image_schedule.")
    if tau_text.shape[:2] != (B, 1):
        raise ValueError(f"tau_text must be [B,1], got {tuple(tau_text.shape)}")

    out_xt: list[list[int]] = []
    out_bags: list[list[list[int]]] = []
    kept_images_list: list[list[torch.Tensor]] = []
    kept_timg_list: list[list[float]] = []

    for b in range(B):
        x1 = x1_ids[b]
        xt = list(xt_list[b])
        bags = [list(bag) for bag in bags_list[b]]

        raw = image_latents_raw[b]
        if raw is None:
            images: list[torch.Tensor] = []
        elif isinstance(raw, list):
            images = raw
        else:
            images = [raw]

        # Validate: num image tokens in x1 must equal provided latents count.
        num_img_tokens_x1 = sum(int(t) == int(image_token_id) for t in x1)
        if num_img_tokens_x1 != len(images):
            raise ValueError(
                f"Mismatch between number of image tokens in x1 ({num_img_tokens_x1}) and "
                f"provided image latents ({len(images)})."
            )

        # Locate image tokens in xt (they were forced-kept during κ-keep step).
        img_pos_xt = [i for i, t in enumerate(xt) if int(t) == int(image_token_id)]
        if len(img_pos_xt) != len(images):
            raise ValueError(
                f"Internal error: expected {len(images)} image tokens in X_t, got {len(img_pos_xt)}."
            )

        tau_text_b = float(tau_text[b, 0].item())
        delete_flags: list[bool] = []
        timgs: list[float] = []

        for _ in images:
            u = torch.rand((), device=device)
            inv = scheduler.kappa_inverse(u)
            inv_f = float(inv.item() if isinstance(inv, torch.Tensor) else inv)
            tau_img = tau_text_b - inv_f
            if tau_img < 0.0:
                delete_flags.append(True)
                timgs.append(0.0)
            else:
                delete_flags.append(False)
                timgs.append(float(min(1.0, tau_img)))

        # Remove deleted images from xt/bags (right-to-left).
        for j in reversed(range(len(images))):
            if not delete_flags[j]:
                continue
            pos = img_pos_xt[j]
            if pos <= 0:
                raise AssertionError("Image token cannot be at position 0 (BOS slot).")
            # Merge: deleted image token goes into previous bag, and bag-after-image merges too.
            bags[pos - 1].append(int(image_token_id))
            bags[pos - 1].extend(bags[pos])
            del bags[pos]
            del xt[pos]

        kept_images: list[torch.Tensor] = []
        kept_timgs: list[float] = []
        for img, is_del, ti in zip(images, delete_flags, timgs):
            if is_del:
                continue
            kept_images.append(img)
            kept_timgs.append(float(ti))

        # Validate remaining image token count matches kept images.
        img_pos_xt_after = [i for i, t in enumerate(xt) if int(t) == int(image_token_id)]
        if len(img_pos_xt_after) != len(kept_images):
            raise ValueError(
                f"After τ_img deletion, expected {len(kept_images)} image tokens in X_t "
                f"but found {len(img_pos_xt_after)}."
            )

        out_xt.append(xt)
        out_bags.append(bags)
        kept_images_list.append(kept_images)
        kept_timg_list.append(kept_timgs)

    return InterleavedImagesOutput(
        xt_list=out_xt,
        bags_list=out_bags,
        kept_images_list=kept_images_list,
        kept_timg_list=kept_timg_list,
    )


def build_unified_train_batch(
    *,
    xt_list: list[list[int]],
    bags_list: list[list[list[int]]],
    kept_images_list: list[list[torch.Tensor]],
    kept_timg_list: list[list[float]],
    t_text: torch.Tensor,
    image_token_id: int,
    pad_id: int,
    dim_latent: int,
    condition_text_on_time: bool,
    device: torch.device,
) -> tuple[UnifiedTrainBatch, torch.Tensor]:
    """
    Build the unified (text+modality-tokens) batch for training.

    Returns:
      (batch, flow_tgt) where flow_tgt is also stored in batch.flow_targets.
    """
    B = len(xt_list)
    zero_lat = torch.zeros((int(dim_latent),), device=device, dtype=torch.float32)

    total_ids_list: list[list[int]] = []
    total_is_mod_list: list[list[bool]] = []
    total_mod_tokens_list: list[list[torch.Tensor]] = []
    total_flow_targets_list: list[list[torch.Tensor]] = []
    total_times_list: list[list[float]] = []
    modality_positions_list: list[list[tuple[int, int, int]]] = []
    xt_to_total_pos_list: list[list[int]] = []

    for b in range(B):
        xt = xt_list[b]
        bags = bags_list[b]
        images = kept_images_list[b]
        timgs = kept_timg_list[b]

        ids: list[int] = []
        is_mod: list[bool] = []
        mod_tokens: list[torch.Tensor] = []
        flow_targets: list[torch.Tensor] = []
        times_b: list[float] = []
        mod_pos: list[tuple[int, int, int]] = []
        xt_to_total: list[int] = []

        t_text_b = float(t_text[b, 0].item())
        t_text_cond = t_text_b if bool(condition_text_on_time) else 0.0
        img_counter = 0

        for tok_id in xt:
            xt_to_total.append(len(ids))
            ids.append(int(tok_id))
            is_mod.append(False)
            mod_tokens.append(zero_lat)
            flow_targets.append(zero_lat)
            times_b.append(float(t_text_cond))

            if int(tok_id) == int(image_token_id) and img_counter < len(images):
                y1 = images[img_counter].to(device=device, dtype=torch.float32)
                ti = float(timgs[img_counter])
                y0 = torch.randn_like(y1)
                yt = ti * y1 + (1.0 - ti) * y0
                flow = y1 - y0

                yt_tok = flatten_latent_to_tokens(yt, dim_latent=int(dim_latent))
                flow_tok = flatten_latent_to_tokens(flow, dim_latent=int(dim_latent))
                n_img = int(yt_tok.shape[0])

                offset = len(ids)
                mod_pos.append((0, offset, n_img))

                for j in range(n_img):
                    ids.append(int(pad_id))
                    is_mod.append(True)
                    mod_tokens.append(yt_tok[j])
                    flow_targets.append(flow_tok[j])
                    times_b.append(float(ti))

                img_counter += 1

        if img_counter != len(images):
            raise ValueError(f"Did not consume all kept images: used {img_counter}, expected {len(images)}.")
        if len(bags) != len(xt_to_total):
            raise AssertionError("bags must align with X_t token positions.")

        total_ids_list.append(ids)
        total_is_mod_list.append(is_mod)
        total_mod_tokens_list.append(mod_tokens)
        total_flow_targets_list.append(flow_targets)
        total_times_list.append(times_b)
        modality_positions_list.append(mod_pos)
        xt_to_total_pos_list.append(xt_to_total)

    x_tok, x_mask = pad_1d(total_ids_list, pad_val=int(pad_id))
    x_tok = x_tok.to(device)
    x_mask = x_mask.to(device)
    Lmax = int(x_tok.shape[1])

    mod_tok = torch.zeros((B, Lmax, int(dim_latent)), device=device, dtype=torch.float32)
    flow_tgt = torch.zeros((B, Lmax, int(dim_latent)), device=device, dtype=torch.float32)
    is_mod_t = torch.zeros((B, Lmax), device=device, dtype=torch.bool)
    times = torch.zeros((B, Lmax), device=device, dtype=torch.float32)

    for b in range(B):
        Lb = len(total_ids_list[b])
        is_mod_t[b, :Lb] = torch.tensor(total_is_mod_list[b], device=device, dtype=torch.bool)
        times[b, :Lb] = torch.tensor(total_times_list[b], device=device, dtype=torch.float32)
        mod_tok[b, :Lb] = torch.stack(total_mod_tokens_list[b], dim=0)
        flow_tgt[b, :Lb] = torch.stack(total_flow_targets_list[b], dim=0)

    Mmax = max((len(m) for m in modality_positions_list), default=0)
    if Mmax > 0:
        mod_pos_tensor = torch.zeros((B, Mmax, 3), device=device, dtype=torch.long)
        for b in range(B):
            for j, (mt, off, ln) in enumerate(modality_positions_list[b]):
                mod_pos_tensor[b, j, 0] = int(mt)
                mod_pos_tensor[b, j, 1] = int(off)
                mod_pos_tensor[b, j, 2] = int(ln)
    else:
        mod_pos_tensor = None

    batch = UnifiedTrainBatch(
        input_ids=x_tok,
        attention_mask=x_mask,
        is_any_modality=is_mod_t,
        modality_tokens=mod_tok,
        flow_targets=flow_tgt,
        times=times,
        modality_positions=mod_pos_tensor,
        xt_to_total_pos_list=xt_to_total_pos_list,
    )
    return batch, flow_tgt


def build_unified_sampler_inputs_bs1(
    *,
    x_list: list[int],
    images: list[dict[str, Any]],
    t_text: float,
    image_token_id: int,
    pad_id: int,
    dim_latent: int,
    condition_text_on_time: bool,
    device: torch.device,
) -> UnifiedSampleInputs:
    """
    Build unified sequence tensors for sampler (bs=1), matching OneFlowSampler's semantics.

    `images` must be aligned to the order of `<|image|>` tokens in x_list.
    Each element is expected to be a dict with:
      - latent: FloatTensor [N, dim_latent]
      - t: float
    """
    ids: list[int] = []
    is_mod: list[bool] = []
    mod_tokens: list[torch.Tensor] = []
    times: list[float] = []

    text_pos_total: list[int] = []
    image_slices: list[tuple[int, int]] = []
    modality_positions: list[tuple[int, int, int]] = []

    zero_lat = torch.zeros((int(dim_latent),), device=device, dtype=torch.float32)
    img_counter = 0
    t_text_cond = float(t_text) if bool(condition_text_on_time) else 0.0

    for tok in x_list:
        text_pos_total.append(len(ids))
        ids.append(int(tok))
        is_mod.append(False)
        mod_tokens.append(zero_lat)
        times.append(float(t_text_cond))

        if int(tok) == int(image_token_id):
            if img_counter >= len(images):
                raise ValueError(
                    f"images list is shorter than number of image tokens in x_list: "
                    f"need idx={img_counter}, have={len(images)}"
                )
            y = images[img_counter]["latent"]
            ti = float(images[img_counter]["t"])
            y = y.to(device=device, dtype=torch.float32)
            n_img = int(y.shape[0])

            start = len(ids)
            modality_positions.append((0, start, n_img))
            for j in range(n_img):
                ids.append(int(pad_id))
                is_mod.append(True)
                mod_tokens.append(y[j])
                times.append(float(ti))
            end = len(ids)
            image_slices.append((start, end))
            img_counter += 1

    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    is_any_modality = torch.tensor([is_mod], device=device, dtype=torch.bool)
    modality_tokens = torch.stack(mod_tokens, dim=0).unsqueeze(0)  # [1,L,d]
    times_tensor = torch.tensor([times], device=device, dtype=torch.float32)
    if modality_positions:
        mod_pos = torch.tensor([modality_positions], device=device, dtype=torch.long)
    else:
        mod_pos = None

    return UnifiedSampleInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        is_any_modality=is_any_modality,
        modality_tokens=modality_tokens,
        times=times_tensor,
        modality_positions=mod_pos,
        text_pos_total=text_pos_total,
        image_slices=image_slices,
    )


