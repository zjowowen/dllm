"""
CPU-only integration test for OneFlow core building blocks.

This test intentionally avoids importing HF Trainer / Transformers to remain runnable in
minimal environments (and to make failures localizable to OneFlow's own logic).

Run:
  pytest -q scripts/tests/test_oneflow_integration_cpu.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.schedulers import LinearKappaScheduler
from dllm.pipelines.oneflow.losses import image_loss_flow_matching, text_loss_paper_eq7
from dllm.pipelines.oneflow.sequence_ops import (
    apply_interleaved_image_schedule,
    build_noised_xt_and_bags,
    build_unified_train_batch,
)
from dllm.pipelines.oneflow.trace import OneFlowTrace, format_unified_position_table


class ToyOneFlowModel(nn.Module):
    """
    A tiny model implementing the OneFlowModel interface for tests.
    """

    def __init__(self, *, vocab_size: int, dim: int, dim_latent: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.dim_latent = int(dim_latent)

        self.text_embed = nn.Embedding(self.vocab_size, self.dim)
        self.lat_proj = nn.Linear(self.dim_latent, self.dim)
        self.time_proj = nn.Linear(1, self.dim)

        self.to_pi = nn.Linear(self.dim, 1)
        self.to_lam = nn.Linear(self.dim, 1)
        self.to_q = nn.Linear(self.dim, self.vocab_size)
        self.to_v = nn.Linear(self.dim, self.dim_latent)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_any_modality: torch.Tensor,
        modality_tokens: torch.Tensor,
        modality_positions=None,
        times: torch.Tensor,
    ):
        # Embedding for text + projected modality embeddings
        ids = input_ids.clamp_min(0)
        text = self.text_embed(ids)
        mod = self.lat_proj(modality_tokens)
        tokens = torch.where(is_any_modality.unsqueeze(-1), mod, text)

        # Add a simple time embedding
        t = times.unsqueeze(-1)  # [B,L,1]
        tokens = tokens + self.time_proj(t)

        pi = torch.sigmoid(self.to_pi(tokens)).squeeze(-1)
        lam = F.softplus(self.to_lam(tokens)).squeeze(-1)
        q_logits = self.to_q(tokens)
        v = self.to_v(tokens)
        return {"pi": pi, "lambda_nonzero": lam, "q_logits": q_logits, "v": v}


def test_cpu_integration_build_forward_loss_backward():
    device = torch.device("cpu")
    torch.manual_seed(0)

    B = 2
    image_token_id = 99
    pad_id = 0
    dim_latent = 4

    # Two samples, one image token each (format mirrors training: BOS ... <image> ... EOS)
    x1_ids = [
        [1, 10, image_token_id, 20, 2],
        [1, 30, 40, image_token_id, 50, 2],
    ]
    # Keep prob 0 => deterministic deletion except BOS + image token forced kept.
    kappa_keep = torch.zeros((B, 1), device=device, dtype=torch.float32)
    noised = build_noised_xt_and_bags(
        x1_ids=x1_ids,
        kappa_keep=kappa_keep,
        device=device,
        image_token_id=image_token_id,
    )

    # Force image kept: τ_text=0.2 and κ^{-1}(u)=0.0 => τ_img=0.2 >=0 => keep, t_img=0.2
    tau_text = torch.tensor([[0.2], [0.2]], device=device, dtype=torch.float32)
    sched = LinearKappaScheduler()
    sched.kappa_inverse = lambda u, **kwargs: torch.zeros_like(u)  # type: ignore[assignment]

    # Two dummy latents (shape [4,2,2] => N=4 modality tokens)
    image_latents_raw = [
        torch.zeros((4, 2, 2), dtype=torch.float32),
        torch.ones((4, 2, 2), dtype=torch.float32),
    ]
    inter = apply_interleaved_image_schedule(
        x1_ids=x1_ids,
        xt_list=noised.xt_list,
        bags_list=noised.bags_list,
        image_latents_raw=image_latents_raw,
        tau_text=tau_text,
        scheduler=sched,
        image_token_id=image_token_id,
        device=device,
    )

    # Condition text on time = False (paper default), so t_text used only for κ-noising, not model times.
    t_text = torch.zeros((B, 1), device=device, dtype=torch.float32)

    unified, _ = build_unified_train_batch(
        xt_list=inter.xt_list,
        bags_list=inter.bags_list,
        kept_images_list=inter.kept_images_list,
        kept_timg_list=inter.kept_timg_list,
        t_text=t_text,
        image_token_id=image_token_id,
        pad_id=pad_id,
        dim_latent=dim_latent,
        condition_text_on_time=False,
        device=device,
    )

    # Tiny model forward
    model = ToyOneFlowModel(vocab_size=128, dim=32, dim_latent=dim_latent).to(device)
    out = model(
        input_ids=unified.input_ids,
        attention_mask=unified.attention_mask,
        is_any_modality=unified.is_any_modality,
        modality_tokens=unified.modality_tokens,
        modality_positions=unified.modality_positions,
        times=unified.times,
    )

    logQ = F.log_softmax(out["q_logits"], dim=-1)
    tl = text_loss_paper_eq7(
        pi=out["pi"],
        lam=out["lambda_nonzero"],
        logQ=logQ,
        bags_list=inter.bags_list,
        xt_positions=unified.xt_to_total_pos_list,
        normalize_by_n=True,
    )
    il = image_loss_flow_matching(
        v=out["v"],
        flow_tgt=unified.flow_targets,
        is_any_modality=unified.is_any_modality,
        normalize_by_tokens=True,
    )

    loss = tl.total + il.loss
    assert torch.isfinite(loss).all()
    loss.backward()

    # Ensure some gradients flowed
    total_grad = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total_grad += float(p.grad.detach().abs().sum().item())
    assert total_grad > 0.0

    # Produce an in-memory trace for docs/debug (no file IO in tests)
    trace = OneFlowTrace(
        seed=0,
        tau_text=tau_text,
        t_text=t_text,
        x1_ids=x1_ids,
        xt_ids=inter.xt_list,
        bags=inter.bags_list,
        unified={
            "input_ids": unified.input_ids,
            "is_any_modality": unified.is_any_modality,
            "times": unified.times,
            "xt_to_total_pos": unified.xt_to_total_pos_list,
            "modality_positions": unified.modality_positions,
        },
        extra={
            "loss_text": float(tl.total.item()),
            "loss_img": float(il.loss.item()),
        },
    )
    s = trace.to_json(indent=2)
    assert "tau_text" in s and "unified" in s

    # sanity-check the position table renderer
    table = format_unified_position_table(
        input_ids=unified.input_ids[0].tolist(),
        is_any_modality=unified.is_any_modality[0].tolist(),
        times=[float(x) for x in unified.times[0].tolist()],
        image_token_id=image_token_id,
        pad_id=pad_id,
    )
    assert "| total_pos | kind |" in table


