from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp_min(1e-12))


@dataclass
class TextEq7Loss:
    total: torch.Tensor
    loss_tok: torch.Tensor
    loss_lam: torch.Tensor
    loss_pi: torch.Tensor


def _flatten_bags_for_gather(
    *,
    bags_list: list[list[list[int]]],
    xt_positions: list[list[int]] | None,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int], list[int]]:
    """
    Helper for Eq(7) loss: build per-sample slot positions and flattened (b, pos, tok_id).

    Returns:
      - pos_lists[b]: list of model positions for each X_t slot i (len = n slots)
      - k_lists[b]: list of k_i (=|A_i|) for each slot i (len = n slots)
      - flat_b, flat_pos, flat_tok: flattened token targets across the batch
    """
    B = len(bags_list)
    pos_lists: list[list[int]] = []
    k_lists: list[list[int]] = []
    flat_b: list[int] = []
    flat_pos: list[int] = []
    flat_tok: list[int] = []

    for b in range(B):
        bags = bags_list[b]
        n = int(len(bags))
        if n <= 0:
            pos_lists.append([])
            k_lists.append([])
            continue

        if xt_positions is None:
            pos = list(range(n))
        else:
            pos = xt_positions[b]
            if len(pos) != n:
                raise ValueError(
                    f"xt_positions length mismatch: got {len(pos)} positions, expected n={n}."
                )
            pos = [int(p) for p in pos]

        ks = [int(len(bag)) for bag in bags]

        # Flatten token CE targets.
        for i, bag in enumerate(bags):
            if not bag:
                continue
            p = int(pos[i])
            for tok in bag:
                flat_b.append(int(b))
                flat_pos.append(int(p))
                flat_tok.append(int(tok))

        pos_lists.append(pos)
        k_lists.append(ks)

    return pos_lists, k_lists, flat_b, flat_pos, flat_tok


def text_loss_paper_eq7_fast(
    *,
    pi: torch.Tensor,
    lam: torch.Tensor,
    logQ: torch.Tensor,
    bags_list: list[list[list[int]]],
    xt_positions: list[list[int]] | None = None,
    normalize_by_n: bool = True,
) -> TextEq7Loss:
    """
    Vectorized paper text loss (Eq. 7) implementation.

    This is functionally equivalent to `text_loss_paper_eq7`, but avoids per-bag tensor
    creation and reduces kernel launch overhead by flattening all bag tokens.
    """
    device = pi.device
    B = int(pi.shape[0])

    from dllm.pipelines.ctmc_utils import pad_1d

    pos_lists, k_lists, flat_b, flat_pos, flat_tok = _flatten_bags_for_gather(
        bags_list=bags_list, xt_positions=xt_positions
    )

    # Pad per-slot positions and k_i to tensors: [B, Lmax]
    pos_pad, slot_mask = pad_1d(pos_lists, pad_val=0)
    k_pad, k_mask = pad_1d(k_lists, pad_val=0)
    # slot_mask and k_mask should match (both represent number of slots); keep slot_mask.
    pos_pad = pos_pad.to(device=device)
    slot_mask_f = slot_mask.to(device=device, dtype=torch.float32)
    k_f = k_pad.to(device=device, dtype=torch.float32)

    # Gather pi/lam at model positions for each slot.
    pi_slots = pi.gather(dim=1, index=pos_pad)
    lam_slots = lam.gather(dim=1, index=pos_pad)

    # π BCE: target 1 if k_i==0 else 0
    pi_slots = pi_slots.clamp(1e-6, 1.0 - 1e-6)
    tgt_zero = (k_f == 0).to(torch.float32)
    bce = F.binary_cross_entropy(pi_slots, tgt_zero, reduction="none")
    loss_pi_sum = (bce * slot_mask_f).sum(dim=1)  # [B]

    # λ_nonzero Poisson on k>0: λ - k log λ
    nz = (k_f > 0).to(torch.float32)
    loss_lam_sum = (((lam_slots - k_f * safe_log(lam_slots)) * nz) * slot_mask_f).sum(dim=1)  # [B]

    # bag-of-tokens CE: -sum_{a in A_i} log Q_i(a)
    if flat_b:
        b_idx = torch.tensor(flat_b, device=device, dtype=torch.long)
        pos_idx = torch.tensor(flat_pos, device=device, dtype=torch.long)
        tok_idx = torch.tensor(flat_tok, device=device, dtype=torch.long)
        # logQ[b, pos, tok]
        tok_logp = logQ[b_idx, pos_idx, tok_idx]
        tok_loss_sum = torch.zeros((B,), device=device, dtype=tok_logp.dtype)
        tok_loss_sum.scatter_add_(0, b_idx, (-tok_logp))
    else:
        tok_loss_sum = pi.new_zeros((B,))

    # Normalize by n slots (per sample) if requested
    n_slots = slot_mask_f.sum(dim=1)  # [B]
    if normalize_by_n:
        denom = n_slots.clamp_min(1.0)
    else:
        denom = torch.ones_like(n_slots)

    loss_tok = (tok_loss_sum / denom).mean()
    loss_pi = (loss_pi_sum / denom).mean()
    loss_lam = (loss_lam_sum / denom).mean()
    total = ((tok_loss_sum + loss_pi_sum + loss_lam_sum) / denom).mean()

    return TextEq7Loss(total=total, loss_tok=loss_tok, loss_lam=loss_lam, loss_pi=loss_pi)


def text_loss_paper_eq7_fast_from_logits(
    *,
    pi: torch.Tensor,
    lam: torch.Tensor,
    q_logits: torch.Tensor,
    bags_list: list[list[list[int]]],
    xt_positions: list[list[int]] | None = None,
    normalize_by_n: bool = True,
) -> TextEq7Loss:
    """
    Fast Eq(7) loss computed directly from logits (avoids materializing full log-softmax).

    Computes token CE as:
      -log softmax(q)[tok] = -(q_tok - logsumexp(q))

    This saves both time and memory compared to building `logQ = log_softmax(q_logits, -1)`.
    """
    device = pi.device
    B = int(pi.shape[0])

    from dllm.pipelines.ctmc_utils import pad_1d

    pos_lists, k_lists, flat_b, flat_pos, flat_tok = _flatten_bags_for_gather(
        bags_list=bags_list, xt_positions=xt_positions
    )

    # Pad per-slot positions and k_i to tensors: [B, Lmax]
    pos_pad, slot_mask = pad_1d(pos_lists, pad_val=0)
    k_pad, _ = pad_1d(k_lists, pad_val=0)
    pos_pad = pos_pad.to(device=device)
    slot_mask_f = slot_mask.to(device=device, dtype=torch.float32)
    k_f = k_pad.to(device=device, dtype=torch.float32)

    # Gather pi/lam at model positions for each slot.
    pi_slots = pi.gather(dim=1, index=pos_pad)
    lam_slots = lam.gather(dim=1, index=pos_pad)

    # π BCE: target 1 if k_i==0 else 0
    pi_slots = pi_slots.clamp(1e-6, 1.0 - 1e-6)
    tgt_zero = (k_f == 0).to(torch.float32)
    bce = F.binary_cross_entropy(pi_slots, tgt_zero, reduction="none")
    loss_pi_sum = (bce * slot_mask_f).sum(dim=1)  # [B]

    # λ_nonzero Poisson on k>0: λ - k log λ
    nz = (k_f > 0).to(torch.float32)
    loss_lam_sum = (((lam_slots - k_f * safe_log(lam_slots)) * nz) * slot_mask_f).sum(dim=1)  # [B]

    # token CE from logits using logsumexp normalization
    if flat_b:
        b_idx = torch.tensor(flat_b, device=device, dtype=torch.long)
        pos_idx = torch.tensor(flat_pos, device=device, dtype=torch.long)
        tok_idx = torch.tensor(flat_tok, device=device, dtype=torch.long)

        # logZ[b, pos] = logsumexp_v q[b,pos,v]  (shape [B, L])
        # NOTE: this is still heavy, but saves the huge [B,L,V] logQ materialization.
        logZ = torch.logsumexp(q_logits, dim=-1)  # [B,L]

        q_tok = q_logits[b_idx, pos_idx, tok_idx]
        logp = q_tok - logZ[b_idx, pos_idx]
        tok_loss_sum = torch.zeros((B,), device=device, dtype=logp.dtype)
        tok_loss_sum.scatter_add_(0, b_idx, (-logp))
    else:
        tok_loss_sum = pi.new_zeros((B,))

    # Normalize by n slots (per sample) if requested
    n_slots = slot_mask_f.sum(dim=1)  # [B]
    if normalize_by_n:
        denom = n_slots.clamp_min(1.0)
    else:
        denom = torch.ones_like(n_slots)

    loss_tok = (tok_loss_sum / denom).mean()
    loss_pi = (loss_pi_sum / denom).mean()
    loss_lam = (loss_lam_sum / denom).mean()
    total = ((tok_loss_sum + loss_pi_sum + loss_lam_sum) / denom).mean()

    return TextEq7Loss(total=total, loss_tok=loss_tok, loss_lam=loss_lam, loss_pi=loss_pi)


def text_loss_paper_eq7(
    *,
    pi: torch.Tensor,
    lam: torch.Tensor,
    logQ: torch.Tensor,
    bags_list: list[list[list[int]]],
    xt_positions: list[list[int]] | None = None,
    normalize_by_n: bool = True,
) -> TextEq7Loss:
    """
    Paper text loss (arXiv:2510.03506 Eq. 7) for a batch.

    Inputs:
    - pi:    [B, L]   (probability of zero insertions at each slot)
    - lam:   [B, L]   (lambda_nonzero at each slot)
    - logQ:  [B, L, V] log-softmax over vocab at each slot
    - bags_list[b][i]: list of token_ids in bag A_i for X_t position i
    - xt_positions[b]: mapping from X_t position i -> position in the model outputs.
      If None, we assume X_t positions map to 0..n-1 (text-only path).

    Note: This implementation normalizes the *sum of all three terms* by n when
    normalize_by_n=True (consistent with existing trainer behavior). This is a
    scaling choice; it does not change the argmin but can affect optimization.
    """
    B = int(pi.shape[0])
    per_total = []
    per_tok = []
    per_lam = []
    per_pi = []

    for b in range(B):
        bags = bags_list[b]
        n = int(len(bags))
        if n <= 0:
            z = pi.new_zeros(())
            per_total.append(z)
            per_tok.append(z)
            per_lam.append(z)
            per_pi.append(z)
            continue

        if xt_positions is None:
            pos_idx = None
            pi_b = pi[b, :n]
            lam_b = lam[b, :n]
        else:
            pos = xt_positions[b]
            if len(pos) != n:
                raise ValueError(
                    f"xt_positions length mismatch: got {len(pos)} positions, expected n={n}."
                )
            pos_idx = torch.tensor(pos, device=pi.device, dtype=torch.long)
            pi_b = pi[b].gather(dim=0, index=pos_idx)
            lam_b = lam[b].gather(dim=0, index=pos_idx)

        # missing counts k_i
        k_list = [len(bag) for bag in bags]
        k_vec = torch.tensor(k_list, device=pi.device, dtype=torch.float32)

        # π BCE: target 1 if k_i==0 else 0
        pi_b = pi_b.clamp(1e-6, 1.0 - 1e-6)
        tgt_zero = (k_vec == 0).to(torch.float32)
        loss_pi = F.binary_cross_entropy(pi_b, tgt_zero, reduction="sum")

        # λ_nonzero Poisson on k>0: λ - k log λ
        nz = (k_vec > 0).to(torch.float32)
        loss_lam = ((lam_b - k_vec * safe_log(lam_b)) * nz).sum()

        # bag-of-tokens CE: -sum_{a in A_i} log Q_i(a)
        loss_tok = pi.new_zeros(())
        for i, bag in enumerate(bags):
            if not bag:
                continue
            tok = torch.tensor(bag, device=pi.device, dtype=torch.long)
            if pos_idx is None:
                pos = i
            else:
                pos = int(pos_idx[i].item())
            loss_tok = loss_tok - logQ[b, pos].gather(dim=-1, index=tok).sum()

        denom = float(n) if normalize_by_n else 1.0
        denom_t = max(1.0, denom)

        per_tok.append(loss_tok / denom_t)
        per_lam.append(loss_lam / denom_t)
        per_pi.append(loss_pi / denom_t)
        per_total.append((loss_tok + loss_lam + loss_pi) / denom_t)

    total = torch.stack(per_total).mean()
    return TextEq7Loss(
        total=total,
        loss_tok=torch.stack(per_tok).mean(),
        loss_lam=torch.stack(per_lam).mean(),
        loss_pi=torch.stack(per_pi).mean(),
    )


@dataclass
class ImageFlowLoss:
    loss: torch.Tensor
    tokens_total: torch.Tensor


def image_loss_flow_matching(
    *,
    v: torch.Tensor,
    flow_tgt: torch.Tensor,
    is_any_modality: torch.Tensor,
    normalize_by_tokens: bool = True,
) -> ImageFlowLoss:
    """
    Flow matching image loss (Eq. 9): ||v(Y_t,t) - (Y1 - Y0)||^2 over modality token positions.
    """
    img_mask = is_any_modality.to(torch.float32)
    tokens_total = img_mask.sum()
    if float(tokens_total.item()) <= 0.0:
        z = v.new_zeros(())
        return ImageFlowLoss(loss=z, tokens_total=tokens_total)

    sq = (v - flow_tgt).pow(2).sum(dim=-1)  # [B,L]
    img_sum = (sq * img_mask).sum()
    if normalize_by_tokens:
        denom = tokens_total.clamp_min(1.0)
        loss = img_sum / denom
    else:
        B = int(v.shape[0])
        loss = img_sum / float(max(1, B))
    return ImageFlowLoss(loss=loss, tokens_total=tokens_total)


