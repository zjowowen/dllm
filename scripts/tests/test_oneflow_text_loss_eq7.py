"""
Unit tests for OneFlow text losses:
  - Paper loss (Eq. 7): token CE + pi BCE + lambda_nonzero zero-truncated Poisson(k>0).
  - Vectorized CTMC loss: survival + positive term, weighted by w(t).

Run:
  pytest -q scripts/tests/test_oneflow_text_loss_eq7.py
"""

import math

import torch

from dllm.pipelines.oneflow.losses import (
    ctmc_loss_vectorized,
    safe_log,
    text_loss_paper_eq7,
    text_loss_paper_eq7_fast,
    text_loss_paper_eq7_fast_from_logits,
)


def _log_softmax_from_probs(probs: list[float]) -> torch.Tensor:
    t = torch.tensor(probs, dtype=torch.float32)
    return torch.log(t / t.sum())


def test_text_loss_eq7_single_sample_expected_value():
    # B=1, L=2, V=5
    pi = torch.tensor([[0.2, 0.9]], dtype=torch.float32)
    lam = torch.tensor([[1.5, 2.0]], dtype=torch.float32)

    # slot0 probs, slot1 probs
    logQ0 = _log_softmax_from_probs([0.1, 0.1, 0.2, 0.3, 0.3])
    logQ1 = _log_softmax_from_probs([0.2, 0.2, 0.2, 0.2, 0.2])
    logQ = torch.stack([logQ0, logQ1], dim=0).unsqueeze(0)  # [1,2,5]

    bags_list = [[[2, 3], []]]  # k0=2, k1=0

    out = text_loss_paper_eq7(
        pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, xt_positions=None, normalize_by_n=True
    )
    out_fast = text_loss_paper_eq7_fast(
        pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, xt_positions=None, normalize_by_n=True
    )
    out_fast_logits = text_loss_paper_eq7_fast_from_logits(
        pi=pi, lam=lam, q_logits=logQ, bags_list=bags_list, xt_positions=None, normalize_by_n=True
    )

    # expected:
    # token CE = -logQ0[2] - logQ0[3]
    exp_tok = -(float(logQ0[2].item()) + float(logQ0[3].item()))
    # pi BCE: y=[0,1] => -log(1-p0) - log(p1)
    exp_pi = -math.log(1.0 - 0.2) - math.log(0.9)
    # lambda zero-truncated Poisson on k>0: lam0 - k0 log lam0 + log(1 - exp(-lam0))
    exp_lam = 1.5 - 2.0 * math.log(1.5) + math.log(1.0 - math.exp(-1.5))
    exp_total = (exp_tok + exp_pi + exp_lam) / 2.0  # normalize by n=2

    assert abs(float(out.loss_tok.item()) - (exp_tok / 2.0)) < 1e-6
    assert abs(float(out.loss_pi.item()) - (exp_pi / 2.0)) < 1e-6
    assert abs(float(out.loss_lam.item()) - (exp_lam / 2.0)) < 1e-6
    assert abs(float(out.total.item()) - exp_total) < 1e-6
    assert abs(float(out_fast.total.item()) - exp_total) < 1e-6
    assert abs(float(out_fast.loss_tok.item()) - float(out.loss_tok.item())) < 1e-6
    assert abs(float(out_fast.loss_pi.item()) - float(out.loss_pi.item())) < 1e-6
    assert abs(float(out_fast.loss_lam.item()) - float(out.loss_lam.item())) < 1e-6
    assert abs(float(out_fast_logits.total.item()) - exp_total) < 1e-6


def test_text_loss_eq7_with_xt_positions_gathers_correct_slots():
    # B=1, L=5, V=4, but only two xt slots at positions [1,4]
    pi = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.9]], dtype=torch.float32)
    lam = torch.tensor([[0.0, 1.5, 0.0, 0.0, 2.0]], dtype=torch.float32)

    # set logQ so that:
    # - at pos=1, token 2 has prob 0.5
    # - at pos=4, token 3 has prob 0.25 (but bag empty there in this test)
    logQ = torch.zeros((1, 5, 4), dtype=torch.float32)
    logQ[0, 1] = _log_softmax_from_probs([0.1, 0.1, 0.5, 0.3])
    logQ[0, 4] = _log_softmax_from_probs([0.25, 0.25, 0.25, 0.25])

    bags_list = [[[2], []]]  # n=2, first slot has one token id=2
    xt_positions = [[1, 4]]

    out = text_loss_paper_eq7(
        pi=pi,
        lam=lam,
        logQ=logQ,
        bags_list=bags_list,
        xt_positions=xt_positions,
        normalize_by_n=True,
    )
    out_fast = text_loss_paper_eq7_fast(
        pi=pi,
        lam=lam,
        logQ=logQ,
        bags_list=bags_list,
        xt_positions=xt_positions,
        normalize_by_n=True,
    )
    out_fast_logits = text_loss_paper_eq7_fast_from_logits(
        pi=pi,
        lam=lam,
        q_logits=logQ,  # already log-softmaxed; logsumexp(logQ)=0
        bags_list=bags_list,
        xt_positions=xt_positions,
        normalize_by_n=True,
    )

    # expected token loss is only from pos=1 token=2
    exp_tok = -float(logQ[0, 1, 2].item())
    exp_pi = -math.log(1.0 - 0.2) - math.log(0.9)  # y=[0,1]
    exp_lam = 1.5 - 1.0 * math.log(1.5) + math.log(1.0 - math.exp(-1.5))
    exp_total = (exp_tok + exp_pi + exp_lam) / 2.0
    assert abs(float(out.total.item()) - exp_total) < 1e-6
    assert abs(float(out_fast.total.item()) - exp_total) < 1e-6
    assert abs(float(out_fast_logits.total.item()) - exp_total) < 1e-6


def test_text_loss_eq7_all_k_zero_has_no_lambda_or_token_loss():
    pi = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
    lam = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    logQ = torch.zeros((1, 2, 3), dtype=torch.float32)  # unused
    bags_list = [[[], []]]

    out = text_loss_paper_eq7(pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, normalize_by_n=True)
    out_fast = text_loss_paper_eq7_fast(pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, normalize_by_n=True)
    out_fast_logits = text_loss_paper_eq7_fast_from_logits(
        pi=pi, lam=lam, q_logits=logQ, bags_list=bags_list, normalize_by_n=True
    )
    assert float(out.loss_tok.item()) == 0.0
    assert float(out.loss_lam.item()) == 0.0
    assert float(out.loss_pi.item()) > 0.0
    assert float(out_fast.loss_tok.item()) == 0.0
    assert float(out_fast.loss_lam.item()) == 0.0
    assert float(out_fast.loss_pi.item()) > 0.0
    assert float(out_fast_logits.loss_tok.item()) == 0.0
    assert float(out_fast_logits.loss_lam.item()) == 0.0
    assert float(out_fast_logits.loss_pi.item()) > 0.0


def test_text_loss_eq7_raises_on_xt_positions_length_mismatch():
    pi = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    lam = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    logQ = torch.zeros((1, 2, 2), dtype=torch.float32)
    bags_list = [[[1], []]]  # n=2
    xt_positions = [[0]]  # mismatch

    try:
        _ = text_loss_paper_eq7(pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, xt_positions=xt_positions)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on xt_positions length mismatch.")

    try:
        _ = text_loss_paper_eq7_fast(pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, xt_positions=xt_positions)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on xt_positions length mismatch (fast).")

    try:
        _ = text_loss_paper_eq7_fast_from_logits(
            pi=pi, lam=lam, q_logits=logQ, bags_list=bags_list, xt_positions=xt_positions
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on xt_positions length mismatch (fast_from_logits).")


def test_text_loss_eq7_fast_matches_reference_random_small():
    torch.manual_seed(0)
    B = 2
    L = 6
    V = 11
    # pi in (0,1), lam > 0
    pi = torch.sigmoid(torch.randn((B, L), dtype=torch.float32))
    lam = torch.nn.functional.softplus(torch.randn((B, L), dtype=torch.float32)) + 1e-3
    logQ = torch.log_softmax(torch.randn((B, L, V), dtype=torch.float32), dim=-1)

    # Build random bags aligned to L slots (no xt_positions).
    bags_list: list[list[list[int]]] = []
    for b in range(B):
        bags = []
        for i in range(L):
            k = int(torch.randint(low=0, high=4, size=(1,)).item())
            bag = [int(torch.randint(low=0, high=V, size=(1,)).item()) for _ in range(k)]
            bags.append(bag)
        bags_list.append(bags)

    ref = text_loss_paper_eq7(pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, xt_positions=None, normalize_by_n=True)
    fast = text_loss_paper_eq7_fast(pi=pi, lam=lam, logQ=logQ, bags_list=bags_list, xt_positions=None, normalize_by_n=True)
    fast_logits = text_loss_paper_eq7_fast_from_logits(
        pi=pi, lam=lam, q_logits=torch.randn((B, L, V), dtype=torch.float32), bags_list=bags_list, xt_positions=None, normalize_by_n=True
    )

    assert abs(float(ref.total.item()) - float(fast.total.item())) < 1e-6
    assert abs(float(ref.loss_tok.item()) - float(fast.loss_tok.item())) < 1e-6
    assert abs(float(ref.loss_pi.item()) - float(fast.loss_pi.item())) < 1e-6
    assert abs(float(ref.loss_lam.item()) - float(fast.loss_lam.item())) < 1e-6

    # For logits-path, compare against reference computed from the same logits.
    q = torch.randn((B, L, V), dtype=torch.float32)
    logQ2 = torch.log_softmax(q, dim=-1)
    ref2 = text_loss_paper_eq7(pi=pi, lam=lam, logQ=logQ2, bags_list=bags_list, xt_positions=None, normalize_by_n=True)
    fast2 = text_loss_paper_eq7_fast_from_logits(pi=pi, lam=lam, q_logits=q, bags_list=bags_list, xt_positions=None, normalize_by_n=True)
    assert abs(float(ref2.total.item()) - float(fast2.total.item())) < 1e-6


# =============================================================================
# CTMC loss vectorized tests
# =============================================================================


def _ctmc_loss_reference_loop(
    lam: torch.Tensor,
    logQ: torch.Tensor,
    bags_list: list[list[list[int]]],
    w: torch.Tensor,
    x1_lengths: list[int],
    xt_positions: list[list[int]] | None = None,
    normalize_by_length: bool = True,
):
    """Reference (loop-based) CTMC loss for correctness verification."""
    B = int(lam.shape[0])
    device = lam.device

    L1 = torch.tensor(x1_lengths, device=device, dtype=torch.float32)
    denom = L1.clamp_min(1.0) if normalize_by_length else torch.ones_like(L1)

    # survival: sum lam at xt positions
    Lambda_hat = torch.zeros((B,), device=device, dtype=torch.float32)
    for b in range(B):
        bags = bags_list[b]
        n = len(bags)
        if xt_positions is None:
            positions = list(range(n))
        else:
            positions = xt_positions[b]
        for pos in positions:
            Lambda_hat[b] = Lambda_hat[b] + lam[b, pos]
    loss_surv = ((w * Lambda_hat) / denom).mean()

    # positive term
    pos_terms = []
    for b in range(B):
        lp = torch.zeros((), dtype=torch.float32, device=device)
        bags = bags_list[b]
        n = len(bags)
        if xt_positions is None:
            positions = list(range(n))
        else:
            positions = xt_positions[b]
        for i in range(n):
            bag = bags[i]
            if not bag:
                continue
            pos = positions[i]
            lp = lp - safe_log(lam[b, pos]) * float(len(bag))
            tok = torch.tensor(bag, device=device, dtype=torch.long)
            lp = lp - logQ[b, pos].gather(dim=-1, index=tok).sum()
        pos_terms.append(lp)
    loss_pos_per = torch.stack(pos_terms)
    loss_pos = ((w * loss_pos_per) / denom).mean()

    return loss_surv + loss_pos, loss_surv, loss_pos


def test_ctmc_loss_vectorized_single_sample():
    """CTMC vectorized matches loop for B=1, no xt_positions."""
    B, L, V = 1, 4, 8
    lam = torch.tensor([[1.5, 0.5, 2.0, 1.0]], dtype=torch.float32)
    logQ = torch.log_softmax(torch.randn((B, L, V), dtype=torch.float32), dim=-1)
    bags_list = [[[2, 3], [], [5], [1, 4, 7]]]
    w = torch.tensor([3.0], dtype=torch.float32)
    x1_lengths = [4]

    ref_total, ref_surv, ref_pos = _ctmc_loss_reference_loop(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w, x1_lengths=x1_lengths,
    )
    vec = ctmc_loss_vectorized(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w, x1_lengths=x1_lengths,
    )
    assert abs(float(vec.total.item()) - float(ref_total.item())) < 1e-5
    assert abs(float(vec.loss_surv.item()) - float(ref_surv.item())) < 1e-5
    assert abs(float(vec.loss_pos.item()) - float(ref_pos.item())) < 1e-5


def test_ctmc_loss_vectorized_batch():
    """CTMC vectorized matches loop for B=3, random bags."""
    torch.manual_seed(42)
    B, L, V = 3, 6, 11
    lam = torch.nn.functional.softplus(torch.randn((B, L))) + 1e-3
    logQ = torch.log_softmax(torch.randn((B, L, V)), dim=-1)
    w = torch.rand((B,)) * 5.0 + 0.1

    bags_list: list[list[list[int]]] = []
    x1_lengths: list[int] = []
    for b in range(B):
        bags = []
        for i in range(L):
            k = int(torch.randint(0, 4, (1,)).item())
            bag = [int(torch.randint(0, V, (1,)).item()) for _ in range(k)]
            bags.append(bag)
        bags_list.append(bags)
        x1_lengths.append(L)

    ref_total, ref_surv, ref_pos = _ctmc_loss_reference_loop(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w, x1_lengths=x1_lengths,
    )
    vec = ctmc_loss_vectorized(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w, x1_lengths=x1_lengths,
    )
    assert abs(float(vec.total.item()) - float(ref_total.item())) < 1e-4
    assert abs(float(vec.loss_surv.item()) - float(ref_surv.item())) < 1e-4
    assert abs(float(vec.loss_pos.item()) - float(ref_pos.item())) < 1e-4


def test_ctmc_loss_vectorized_with_xt_positions():
    """CTMC vectorized matches loop when xt_positions is provided (mixed-modal path)."""
    B, L, V = 2, 8, 5
    lam = torch.nn.functional.softplus(torch.randn((B, L))) + 1e-3
    logQ = torch.log_softmax(torch.randn((B, L, V)), dim=-1)
    w = torch.tensor([2.0, 4.0])
    # xt_positions maps 3 xt slots to specific model positions
    xt_positions = [[1, 3, 6], [0, 4, 7]]
    bags_list = [
        [[1, 2], [], [3]],       # 3 slots for sample 0
        [[4], [0, 1], []],       # 3 slots for sample 1
    ]
    x1_lengths = [3, 3]

    ref_total, ref_surv, ref_pos = _ctmc_loss_reference_loop(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w,
        x1_lengths=x1_lengths, xt_positions=xt_positions,
    )
    vec = ctmc_loss_vectorized(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w,
        x1_lengths=x1_lengths, xt_positions=xt_positions,
    )
    assert abs(float(vec.total.item()) - float(ref_total.item())) < 1e-5
    assert abs(float(vec.loss_surv.item()) - float(ref_surv.item())) < 1e-5
    assert abs(float(vec.loss_pos.item()) - float(ref_pos.item())) < 1e-5


def test_ctmc_loss_vectorized_all_empty_bags():
    """CTMC vectorized handles all-empty bags correctly (pos_terms = 0)."""
    B, L, V = 2, 3, 4
    lam = torch.ones((B, L))
    logQ = torch.zeros((B, L, V))
    w = torch.tensor([1.0, 1.0])
    bags_list = [[[], [], []], [[], [], []]]
    x1_lengths = [3, 3]

    vec = ctmc_loss_vectorized(
        lam=lam, logQ=logQ, bags_list=bags_list, w=w, x1_lengths=x1_lengths,
    )
    # With all empty bags, pos_terms should be 0, loss_pos should be 0.
    assert abs(float(vec.loss_pos.item())) < 1e-6
    # Survival is w * sum(lam) / denom = 1 * 3 / 3 = 1.0
    assert abs(float(vec.loss_surv.item()) - 1.0) < 1e-5


