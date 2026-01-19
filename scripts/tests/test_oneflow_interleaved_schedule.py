"""
Unit tests for OneFlow interleaved image schedule (Algorithm 3, τ_img deletion/keep).

Run:
  pytest -q scripts/tests/test_oneflow_interleaved_schedule.py
"""

import torch

from dllm.core.schedulers import LinearKappaScheduler
from dllm.pipelines.oneflow.sequence_ops import (
    apply_interleaved_image_schedule,
    build_noised_xt_and_bags,
)


def test_interleaved_schedule_deletes_image_and_merges_bags():
    device = torch.device("cpu")
    image_token_id = 99

    # x1: BOS, t1, <img>, t2, t3
    x1_ids = [[1, 2, image_token_id, 3, 4]]
    # κ-keep=0 => only BOS and <img> survive (image token forced kept)
    kappa_keep = torch.zeros((1, 1), device=device)
    noised = build_noised_xt_and_bags(
        x1_ids=x1_ids, kappa_keep=kappa_keep, device=device, image_token_id=image_token_id
    )
    assert noised.xt_list == [[1, image_token_id]]
    assert noised.bags_list == [[[2], [3, 4]]]

    # Force delete: τ_text small and κ^{-1}(u) large => τ_img < 0
    sched = LinearKappaScheduler()

    def inv_const(u, **kwargs):
        return torch.ones_like(u) * 0.9

    sched.kappa_inverse = inv_const  # type: ignore[assignment]

    tau_text = torch.tensor([[0.2]], device=device, dtype=torch.float32)
    image_latents_raw = [torch.zeros((4, 2, 2), dtype=torch.float32)]

    out = apply_interleaved_image_schedule(
        x1_ids=x1_ids,
        xt_list=noised.xt_list,
        bags_list=noised.bags_list,
        image_latents_raw=image_latents_raw,
        tau_text=tau_text,
        scheduler=sched,
        image_token_id=image_token_id,
        device=device,
    )

    # Image token is deleted, merged into previous bag, and bag-after-image is merged too.
    assert out.xt_list == [[1]]
    assert out.bags_list == [[[2, image_token_id, 3, 4]]]
    assert out.kept_images_list == [[]]
    assert out.kept_timg_list == [[]]


def test_interleaved_schedule_keeps_image_and_sets_t_img():
    device = torch.device("cpu")
    image_token_id = 99
    x1_ids = [[1, 2, image_token_id, 3, 4]]
    kappa_keep = torch.zeros((1, 1), device=device)
    noised = build_noised_xt_and_bags(
        x1_ids=x1_ids, kappa_keep=kappa_keep, device=device, image_token_id=image_token_id
    )

    # Force keep: κ^{-1}(u) small => τ_img >= 0 and t_img = τ_img
    sched = LinearKappaScheduler()

    def inv_const(u, **kwargs):
        return torch.ones_like(u) * 0.1

    sched.kappa_inverse = inv_const  # type: ignore[assignment]

    tau_text = torch.tensor([[0.2]], device=device, dtype=torch.float32)
    y1 = torch.zeros((4, 2, 2), dtype=torch.float32)
    image_latents_raw = [y1]

    out = apply_interleaved_image_schedule(
        x1_ids=x1_ids,
        xt_list=noised.xt_list,
        bags_list=noised.bags_list,
        image_latents_raw=image_latents_raw,
        tau_text=tau_text,
        scheduler=sched,
        image_token_id=image_token_id,
        device=device,
    )

    assert out.xt_list == [[1, image_token_id]]
    assert out.bags_list == [[[2], [3, 4]]]
    assert len(out.kept_images_list[0]) == 1
    assert torch.equal(out.kept_images_list[0][0], y1)
    assert len(out.kept_timg_list[0]) == 1
    assert abs(float(out.kept_timg_list[0][0]) - 0.1) < 1e-6


def test_interleaved_schedule_raises_on_image_count_mismatch():
    device = torch.device("cpu")
    image_token_id = 99
    x1_ids = [[1, image_token_id, 2]]
    kappa_keep = torch.ones((1, 1), device=device)
    noised = build_noised_xt_and_bags(
        x1_ids=x1_ids, kappa_keep=kappa_keep, device=device, image_token_id=image_token_id
    )

    sched = LinearKappaScheduler()
    tau_text = torch.tensor([[1.2]], device=device, dtype=torch.float32)
    # Mismatch: x1 has 1 image token but we provide 0 latents
    image_latents_raw = [None]

    try:
        _ = apply_interleaved_image_schedule(
            x1_ids=x1_ids,
            xt_list=noised.xt_list,
            bags_list=noised.bags_list,
            image_latents_raw=image_latents_raw,
            tau_text=tau_text,
            scheduler=sched,
            image_token_id=image_token_id,
            device=device,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on image token / latent count mismatch")


