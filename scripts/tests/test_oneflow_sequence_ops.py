"""
Unit tests for OneFlow sequence operations (noising + unified sequence build).

Run:
  pytest -q scripts/tests/test_oneflow_sequence_ops.py
"""

import torch

from dllm.pipelines.oneflow.sequence_ops import (
    build_noised_xt_and_bags,
    build_unified_train_batch,
)


def test_build_noised_xt_and_bags_kappa_keep_one_keeps_all():
    device = torch.device("cpu")
    x1_ids = [[1, 2, 3, 4]]
    kappa_keep = torch.ones((1, 1), device=device)

    out = build_noised_xt_and_bags(x1_ids=x1_ids, kappa_keep=kappa_keep, device=device)
    assert out.xt_list == [[1, 2, 3, 4]]
    assert out.bags_list == [[[], [], [], []]]
    assert out.keep_mask_list == [[True, True, True, True]]


def test_build_noised_xt_and_bags_kappa_keep_zero_keeps_only_bos():
    device = torch.device("cpu")
    x1_ids = [[10, 11, 12, 13]]
    kappa_keep = torch.zeros((1, 1), device=device)

    torch.manual_seed(0)
    out = build_noised_xt_and_bags(x1_ids=x1_ids, kappa_keep=kappa_keep, device=device)
    assert out.xt_list == [[10]]
    assert out.bags_list == [[[11, 12, 13]]]
    assert out.keep_mask_list[0][0] is True
    assert sum(out.keep_mask_list[0]) == 1


def test_build_noised_xt_and_bags_prompt_len_forces_prefix_keep():
    device = torch.device("cpu")
    # BOS, p1, t2, t3
    x1_ids = [[1, 100, 200, 300]]
    kappa_keep = torch.zeros((1, 1), device=device)

    torch.manual_seed(0)
    out = build_noised_xt_and_bags(
        x1_ids=x1_ids, kappa_keep=kappa_keep, device=device, prompt_len_list=[2]
    )
    assert out.xt_list == [[1, 100]]
    assert out.bags_list == [[[], [200, 300]]]
    assert out.keep_mask_list == [[True, True, False, False]]


def test_build_noised_xt_and_bags_image_token_is_forced_kept():
    device = torch.device("cpu")
    image_token_id = 99
    # BOS, t1, <image>, t2
    x1_ids = [[1, 2, image_token_id, 3]]
    kappa_keep = torch.zeros((1, 1), device=device)

    torch.manual_seed(0)
    out = build_noised_xt_and_bags(
        x1_ids=x1_ids, kappa_keep=kappa_keep, device=device, image_token_id=image_token_id
    )
    assert out.xt_list == [[1, image_token_id]]
    assert out.bags_list == [[[2], [3]]]
    assert out.keep_mask_list == [[True, False, True, False]]


def test_build_noised_xt_and_bags_prompt_len_cannot_span_image_token():
    device = torch.device("cpu")
    image_token_id = 99
    # BOS, t1, <image>, t2
    x1_ids = [[1, 2, image_token_id, 3]]
    kappa_keep = torch.ones((1, 1), device=device)

    try:
        _ = build_noised_xt_and_bags(
            x1_ids=x1_ids,
            kappa_keep=kappa_keep,
            device=device,
            prompt_len_list=[3],
            image_token_id=image_token_id,
            disallow_image_in_prompt=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when prompt_len spans over image token.")


def test_build_unified_train_batch_shapes_and_positions_single_image():
    device = torch.device("cpu")
    pad_id = 0
    dim_latent = 4
    image_token_id = 99

    # X_t contains one image anchor token.
    xt_list = [[1, image_token_id]]
    bags_list = [[[], []]]
    # one image latent: [4,H,W] where H=W=2 => N=4 tokens
    y1 = torch.zeros((4, 2, 2), dtype=torch.float32)
    kept_images_list = [[y1]]
    kept_timg_list = [[0.3]]
    t_text = torch.tensor([[0.7]], device=device, dtype=torch.float32)

    torch.manual_seed(0)
    batch, _ = build_unified_train_batch(
        xt_list=xt_list,
        bags_list=bags_list,
        kept_images_list=kept_images_list,
        kept_timg_list=kept_timg_list,
        t_text=t_text,
        image_token_id=image_token_id,
        pad_id=pad_id,
        dim_latent=dim_latent,
        condition_text_on_time=False,
        device=device,
    )

    assert batch.input_ids.shape == (1, 6)
    assert batch.attention_mask.shape == (1, 6)
    assert batch.is_any_modality.shape == (1, 6)
    assert batch.modality_tokens.shape == (1, 6, 4)
    assert batch.flow_targets.shape == (1, 6, 4)
    assert batch.times.shape == (1, 6)

    # Mapping from X_t positions to unified positions should be [0,1]
    assert batch.xt_to_total_pos_list == [[0, 1]]

    # Expect modality block immediately after the image token position.
    assert batch.modality_positions is not None
    assert batch.modality_positions.shape == (1, 1, 3)
    assert batch.modality_positions[0, 0].tolist() == [0, 2, 4]

    # is_any_modality should be true only on [2,6)
    assert batch.is_any_modality[0].tolist() == [False, False, True, True, True, True]

    # times: text positions use constant 0.0 when condition_text_on_time=False
    assert batch.times[0, 0].item() == 0.0
    assert batch.times[0, 1].item() == 0.0
    # modality positions use t_img=0.3
    assert torch.allclose(batch.times[0, 2:], torch.full((4,), 0.3))


def test_build_unified_train_batch_padding_for_samples_without_images():
    device = torch.device("cpu")
    pad_id = 0
    dim_latent = 4
    image_token_id = 99

    xt_list = [
        [1, image_token_id],  # with image
        [1],  # no image
    ]
    bags_list = [
        [[], []],
        [[]],
    ]
    y1 = torch.zeros((4, 2, 2), dtype=torch.float32)
    kept_images_list = [[y1], []]
    kept_timg_list = [[0.5], []]
    t_text = torch.tensor([[0.2], [0.9]], device=device, dtype=torch.float32)

    torch.manual_seed(0)
    batch, _ = build_unified_train_batch(
        xt_list=xt_list,
        bags_list=bags_list,
        kept_images_list=kept_images_list,
        kept_timg_list=kept_timg_list,
        t_text=t_text,
        image_token_id=image_token_id,
        pad_id=pad_id,
        dim_latent=dim_latent,
        condition_text_on_time=False,
        device=device,
    )

    assert batch.input_ids.shape == (2, 6)
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert batch.attention_mask[1].tolist() == [1, 0, 0, 0, 0, 0]
    assert batch.is_any_modality[1].tolist() == [False, False, False, False, False, False]
    # second row has no modalities, so its modality_positions row should be zeros (padded)
    assert batch.modality_positions is not None
    assert batch.modality_positions.shape == (2, 1, 3)
    assert batch.modality_positions[1, 0].tolist() == [0, 0, 0]


