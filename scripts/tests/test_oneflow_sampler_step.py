"""
Unit tests for sampler-step core logic:
- p^λ and p^π formulas + clamping
- right-to-left insertion semantics
- image token insertion aligns images list with image token order
- unified sampler input concatenation (bs=1) produces consistent slices

Run:
  pytest -q scripts/tests/test_oneflow_sampler_step.py
"""

import torch

from dllm.pipelines.oneflow.sampler_ops import apply_insertions_right_to_left, p_lam, p_pi
from dllm.pipelines.oneflow.sequence_ops import build_unified_sampler_inputs_bs1


def test_p_lam_formula_and_clamp():
    # dt*w*lam = 0.1*2*3 = 0.6
    assert abs(p_lam(dt_text=0.1, w=2.0, lam_nonzero=3.0) - 0.6) < 1e-9
    # negative -> 0
    assert p_lam(dt_text=0.1, w=-2.0, lam_nonzero=3.0) == 0.0
    # large -> clamp to <1
    assert 0.999 <= p_lam(dt_text=1.0, w=10.0, lam_nonzero=10.0) < 1.0


def test_p_pi_formula_and_clamp():
    assert abs(p_pi(pi=0.2) - 0.8) < 1e-9
    assert p_pi(pi=1.2) == 0.0
    assert 0.999 <= p_pi(pi=-1.0) < 1.0


def test_apply_insertions_right_to_left_keeps_slot_semantics():
    image_token_id = 99
    x = [1, 2, 3]
    images = []

    # Insert 9 after slot 0 (after token 1), and 8 after slot 1 (after token 2 in original x).
    insertions = [(0, 9), (1, 8)]

    apply_insertions_right_to_left(
        x_list=x,
        insertions=insertions,
        image_token_id=image_token_id,
        images=images,
        make_new_image=lambda: {"latent": "NEW", "t": 0.0},
    )
    assert x == [1, 9, 2, 8, 3]
    assert images == []


def test_apply_insertions_image_alignment_inserts_before_existing_image():
    image_token_id = 99
    # One existing image token in the sequence.
    x = [1, 10, image_token_id, 20]
    images = [{"latent": "OLD0", "t": 0.0}]

    # Insert a new image token after token 10 (slot index 1), i.e. before the existing image token.
    insertions = [(1, image_token_id)]
    apply_insertions_right_to_left(
        x_list=x,
        insertions=insertions,
        image_token_id=image_token_id,
        images=images,
        make_new_image=lambda: {"latent": "NEW0", "t": 0.0},
    )
    assert x == [1, 10, image_token_id, image_token_id, 20]
    assert [im["latent"] for im in images] == ["NEW0", "OLD0"]


def test_apply_insertions_image_alignment_inserts_after_existing_image():
    image_token_id = 99
    x = [1, image_token_id, 10, 20]
    images = [{"latent": "OLD0", "t": 0.0}]

    # Insert new image after token 10 (slot index 2), i.e. after the existing image token.
    insertions = [(2, image_token_id)]
    apply_insertions_right_to_left(
        x_list=x,
        insertions=insertions,
        image_token_id=image_token_id,
        images=images,
        make_new_image=lambda: {"latent": "NEW1", "t": 0.0},
    )
    assert x == [1, image_token_id, 10, image_token_id, 20]
    assert [im["latent"] for im in images] == ["OLD0", "NEW1"]


def test_build_unified_sampler_inputs_bs1_shapes_and_slices():
    device = torch.device("cpu")
    pad_id = 0
    dim_latent = 4
    image_token_id = 99

    x_list = [1, 10, image_token_id, 20]
    images = [
        {
            "latent": torch.zeros((4, dim_latent), dtype=torch.float32),
            "t": 0.3,
        }
    ]

    uni = build_unified_sampler_inputs_bs1(
        x_list=x_list,
        images=images,
        t_text=0.7,
        image_token_id=image_token_id,
        pad_id=pad_id,
        dim_latent=dim_latent,
        condition_text_on_time=False,
        device=device,
    )

    # L = len(text)=4 plus N=4 modality tokens => 8
    assert uni.input_ids.shape == (1, 8)
    assert uni.is_any_modality.shape == (1, 8)
    assert uni.modality_tokens.shape == (1, 8, 4)
    assert uni.times.shape == (1, 8)
    # The last text token comes AFTER the modality block, so its unified position is shifted.
    assert uni.text_pos_total == [0, 1, 2, 7]
    assert uni.image_slices == [(3, 7)]  # modality block starts after image token at pos=2
    assert uni.modality_positions is not None
    assert uni.modality_positions.shape == (1, 1, 3)
    assert uni.modality_positions[0, 0].tolist() == [0, 3, 4]

    # times: text positions are constant 0, modality positions use t_img=0.3
    assert uni.times[0, 0].item() == 0.0
    assert uni.times[0, 1].item() == 0.0
    assert uni.times[0, 2].item() == 0.0
    assert abs(float(uni.times[0, 3].item()) - 0.3) < 1e-6


