from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def clamp_prob(p: float, *, eps: float = 1e-6) -> float:
    """Clamp probability into [0, 1-eps] to avoid bernoulli edge cases."""
    p = float(p)
    if p <= 0.0:
        return 0.0
    hi = 1.0 - float(eps)
    if p >= hi:
        return hi
    return p


def p_lam(*, dt_text: float, w: float, lam_nonzero: float) -> float:
    """Algorithm 2: p^λ = Δt * (κ'(t)/(1-κ(t))) * λ_nonzero."""
    return clamp_prob(float(dt_text) * float(w) * float(lam_nonzero))


def p_pi(*, pi: float) -> float:
    """Algorithm 2: p^π = 1 - π."""
    return clamp_prob(1.0 - float(pi))


@dataclass
class ImageState:
    latent: Any
    t: float


def apply_insertions_right_to_left(
    *,
    x_list: list[int],
    insertions: list[tuple[int, int]],
    image_token_id: int,
    images: list[dict[str, Any]],
    make_new_image: Callable[[], dict[str, Any]],
) -> None:
    """
    Apply (slot_index_in_x, token_id) insertions in-place from right-to-left.

    This mirrors the sampler behavior:
    - insertion at slot i happens at position (i + 1) in the token list.
    - if inserting an image token, also insert a new image state into `images`
      aligned to the order of image tokens in the *updated* x_list.
    """
    # Ensure stable index semantics when multiple insertions are applied.
    for i, a in sorted(insertions, key=lambda x: x[0], reverse=True):
        insert_at = int(i) + 1
        x_list.insert(insert_at, int(a))

        if int(a) == int(image_token_id):
            # Align new image to the order of image tokens in x_list:
            # insert at the number of image tokens strictly BEFORE insert_at.
            img_insert_index = sum(
                1 for t in x_list[:insert_at] if int(t) == int(image_token_id)
            )
            images.insert(int(img_insert_index), make_new_image())


