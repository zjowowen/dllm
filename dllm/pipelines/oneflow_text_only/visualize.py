from __future__ import annotations

import colorsys
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    Console = None
    Panel = None
    Table = None
    Text = None
    RICH_AVAILABLE = False


def _ensure_rich_available() -> None:
    if not RICH_AVAILABLE:
        raise ImportError(
            "Visualization requested but `rich` is not installed. "
            "Please install project dependencies including `rich`."
        )


def build_intermediates_from_histories(
    *,
    histories: Sequence[Tensor],
    pad_token_id: int,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, list[int]]:
    """
    Convert variable-length sampler histories into a padded timeline tensor.

    Returns:
      - intermediates: [T, 1, Lmax]
      - time_grid: [T] linearly spaced in [0,1]
      - valid_lengths: list[int] with original sequence length for each step
    """
    if not histories:
        raise ValueError("histories is empty; cannot build visual intermediates.")

    seqs: list[Tensor] = []
    lengths: list[int] = []
    for h in histories:
        t = h.detach()
        if t.dim() == 2:
            if t.shape[0] != 1:
                raise ValueError(f"Expected history tensor [1, L], got {tuple(t.shape)}")
            t = t[0]
        elif t.dim() != 1:
            raise ValueError(f"Expected history tensor [L] or [1, L], got {tuple(t.shape)}")
        t = t.to(dtype=torch.long, device="cpu")
        seqs.append(t)
        lengths.append(int(t.numel()))

    lmax = max(lengths)
    t_steps = len(seqs)
    target_device = device if device is not None else torch.device("cpu")
    intermediates = torch.full(
        (t_steps, 1, lmax),
        fill_value=int(pad_token_id),
        dtype=torch.long,
        device=target_device,
    )
    for i, seq in enumerate(seqs):
        cur = seq.to(device=target_device)
        intermediates[i, 0, : cur.numel()] = cur

    time_grid = torch.linspace(0.0, 1.0, steps=t_steps, device=target_device, dtype=torch.float32)
    return intermediates, time_grid, lengths


def compute_token_order(intermediates: Tensor) -> Tensor:
    if intermediates.shape[0] <= 1:
        return torch.zeros_like(intermediates[0], dtype=torch.float32)

    changes = intermediates[1:].ne(intermediates[:-1]).to(torch.float32)
    n_steps = intermediates.shape[0]
    change_step_idx = (
        torch.arange(1, n_steps, device=intermediates.device, dtype=torch.float32).view(-1, 1, 1).expand_as(changes)
    )
    return (changes * change_step_idx).amax(dim=0)


def _decode_token(tokenizer: PreTrainedTokenizer, token_id: int) -> str:
    token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    return token if token else "<empty>"


def _progress_bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def _clamp01(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _interpolate_rgb(
    alpha: float,
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
) -> tuple[int, int, int]:
    alpha = _clamp01(alpha)
    r = int(round(start_rgb[0] + alpha * (end_rgb[0] - start_rgb[0])))
    g = int(round(start_rgb[1] + alpha * (end_rgb[1] - start_rgb[1])))
    b = int(round(start_rgb[2] + alpha * (end_rgb[2] - start_rgb[2])))
    return r, g, b


def _piecewise_colormap(
    alpha: float,
    points: tuple[tuple[float, tuple[int, int, int]], ...],
) -> tuple[int, int, int]:
    alpha = _clamp01(alpha)
    if alpha <= points[0][0]:
        return points[0][1]
    if alpha >= points[-1][0]:
        return points[-1][1]

    for idx in range(1, len(points)):
        left_t, left_rgb = points[idx - 1]
        right_t, right_rgb = points[idx]
        if alpha <= right_t:
            local_alpha = (alpha - left_t) / max(right_t - left_t, 1e-12)
            return _interpolate_rgb(local_alpha, left_rgb, right_rgb)

    return points[-1][1]


def _jet_rgb(alpha: float) -> tuple[int, int, int]:
    alpha = _clamp01(alpha)
    r = _clamp01(1.5 - abs(4.0 * alpha - 3.0))
    g = _clamp01(1.5 - abs(4.0 * alpha - 2.0))
    b = _clamp01(1.5 - abs(4.0 * alpha - 1.0))
    return int(round(255 * r)), int(round(255 * g)), int(round(255 * b))


def _rainbow_rgb(alpha: float) -> tuple[int, int, int]:
    alpha = _clamp01(alpha)
    hue = (1.0 - alpha) * 0.75
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(round(255 * r)), int(round(255 * g)), int(round(255 * b))


def _viridis_rgb(alpha: float) -> tuple[int, int, int]:
    points = (
        (0.00, (68, 1, 84)),
        (0.13, (71, 44, 122)),
        (0.25, (59, 81, 139)),
        (0.38, (44, 113, 142)),
        (0.50, (33, 144, 141)),
        (0.63, (39, 173, 129)),
        (0.75, (92, 200, 99)),
        (0.88, (170, 220, 50)),
        (1.00, (253, 231, 37)),
    )
    return _piecewise_colormap(alpha, points)


def _rgb_from_alpha(alpha: float, color_map: str) -> tuple[int, int, int]:
    cmap = color_map.lower()
    if cmap == "rainbow":
        return _rainbow_rgb(alpha)
    if cmap == "viridis":
        return _viridis_rgb(alpha)
    if cmap == "black_to_lightblue":
        return _interpolate_rgb(alpha, start_rgb=(0, 0, 0), end_rgb=(173, 216, 230))
    return _jet_rgb(alpha)


def _build_color_style(order_id: float, n_time_steps: int, color_map: str) -> str:
    denom = max(n_time_steps - 1, 1)
    alpha = _clamp01(float(order_id) / float(denom))
    r, g, b = _rgb_from_alpha(alpha=alpha, color_map=color_map)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_colorbar_text(color_map: str, width: int = 56) -> Text:
    _ensure_rich_available()
    bar = Text("t=0 ", style="bold")
    for idx in range(max(width, 2)):
        alpha = idx / max(width - 1, 1)
        r, g, b = _rgb_from_alpha(alpha=alpha, color_map=color_map)
        bar.append("█", style=f"#{r:02x}{g:02x}{b:02x}")
    bar.append(" t=1", style="bold")
    return bar


def _build_token_text(
    *,
    tokenizer: PreTrainedTokenizer,
    token_ids: Tensor,
    token_order: Tensor,
    n_time_steps: int,
    color_map: str,
    valid_length: int,
) -> Text:
    _ensure_rich_available()
    txt = Text()
    n = int(max(0, min(valid_length, int(token_ids.shape[0]))))
    for idx in range(n):
        token_id = int(token_ids[idx].item())
        order_id = float(token_order[idx].item())
        tok = _decode_token(tokenizer=tokenizer, token_id=token_id)
        txt.append(tok, style=_build_color_style(order_id, n_time_steps, color_map))
    return txt


def _build_timeline_table(
    *,
    intermediates: Tensor,
    time_grid: Tensor,
    sample_idx: int,
    valid_lengths: Sequence[int],
) -> Table:
    _ensure_rich_available()
    table = Table(title=f"Sample {sample_idx}: transition timeline", show_header=True, header_style="bold")
    table.add_column("t")
    table.add_column("progress")
    table.add_column("seq_len")
    table.add_column("changed_tokens")

    n_steps = intermediates.shape[0]
    for step_idx in range(n_steps):
        progress = step_idx / max(n_steps - 1, 1)
        if step_idx == 0:
            changed = 0
        else:
            prev_state = intermediates[step_idx - 1, sample_idx]
            cur_state = intermediates[step_idx, sample_idx]
            changed = int(prev_state.ne(cur_state).sum().item())
        table.add_row(
            f"{float(time_grid[step_idx].item()):.3f}",
            _progress_bar(progress),
            str(int(valid_lengths[step_idx])),
            str(changed),
        )
    return table


def _render_terminal_replay(
    *,
    console: Console,
    intermediates: Tensor,
    time_grid: Tensor,
    tokenizer: PreTrainedTokenizer,
    sample_idx: int,
    valid_lengths: Sequence[int],
    color_map: str,
    replay_step_delay_sec: float,
    replay_hold_sec: float,
    replay_until_interrupt: bool,
) -> None:
    token_order = compute_token_order(intermediates=intermediates)
    n_time_steps = intermediates.shape[0]
    replay_step_delay_sec = max(0.0, float(replay_step_delay_sec))
    replay_hold_sec = max(0.0, float(replay_hold_sec))
    iteration = 0

    while True:
        for step_idx in range(n_time_steps):
            token_ids = intermediates[step_idx, sample_idx]
            per_token_order = token_order[sample_idx]
            token_text = _build_token_text(
                tokenizer=tokenizer,
                token_ids=token_ids,
                token_order=per_token_order,
                n_time_steps=n_time_steps,
                color_map=color_map,
                valid_length=int(valid_lengths[step_idx]),
            )
            if step_idx == 0:
                changed_tokens = 0
            else:
                prev_state = intermediates[step_idx - 1, sample_idx]
                cur_state = intermediates[step_idx, sample_idx]
                changed_tokens = int(prev_state.ne(cur_state).sum().item())

            console.clear()
            console.print(Panel(_build_colorbar_text(color_map=color_map), title=f"Colorbar ({color_map})"))
            console.print(
                Panel(
                    token_text,
                    title=f"Visual sample {sample_idx} (replay {iteration + 1}, colormap={color_map})",
                )
            )
            table = Table(title=f"Replay step {step_idx + 1}/{n_time_steps}", show_header=True, header_style="bold")
            table.add_column("t")
            table.add_column("progress")
            table.add_column("seq_len")
            table.add_column("changed_tokens")
            table.add_row(
                f"{float(time_grid[step_idx].item()):.3f}",
                _progress_bar(step_idx / max(n_time_steps - 1, 1)),
                str(int(valid_lengths[step_idx])),
                str(changed_tokens),
            )
            console.print(table)
            if step_idx == n_time_steps - 1:
                time.sleep(replay_hold_sec)
            else:
                time.sleep(replay_step_delay_sec)

        if not replay_until_interrupt:
            break
        iteration += 1


def render_rich_timeline(
    *,
    intermediates: Tensor,
    time_grid: Tensor,
    tokenizer: PreTrainedTokenizer,
    sample_indices: Sequence[int],
    valid_lengths: Sequence[int],
    color_map: str = "jet",
    terminal_replay: bool = False,
    replay_step_delay_sec: float = 0.25,
    replay_hold_sec: float = 5.0,
    replay_until_interrupt: bool = True,
    console: Optional[Console] = None,
) -> None:
    _ensure_rich_available()
    console = console or Console(color_system="truecolor", force_terminal=True)
    use_replay = bool(terminal_replay) and (not getattr(console, "record", False))

    if use_replay:
        for sample_idx in sample_indices:
            _render_terminal_replay(
                console=console,
                intermediates=intermediates,
                time_grid=time_grid,
                tokenizer=tokenizer,
                sample_idx=int(sample_idx),
                valid_lengths=valid_lengths,
                color_map=color_map,
                replay_step_delay_sec=replay_step_delay_sec,
                replay_hold_sec=replay_hold_sec,
                replay_until_interrupt=replay_until_interrupt,
            )
        return

    token_order = compute_token_order(intermediates=intermediates)
    n_time_steps = intermediates.shape[0]
    for sample_idx in sample_indices:
        console.print(Panel(_build_colorbar_text(color_map=color_map), title=f"Colorbar ({color_map})"))
        final_ids = intermediates[-1, sample_idx]
        per_token_order = token_order[sample_idx]
        token_text = _build_token_text(
            tokenizer=tokenizer,
            token_ids=final_ids,
            token_order=per_token_order,
            n_time_steps=n_time_steps,
            color_map=color_map,
            valid_length=int(valid_lengths[-1]),
        )
        console.print(Panel(token_text, title=f"Visual sample {sample_idx} (colormap={color_map}, earlier->later)"))
        console.print(
            _build_timeline_table(
                intermediates=intermediates,
                time_grid=time_grid,
                sample_idx=int(sample_idx),
                valid_lengths=valid_lengths,
            )
        )


def save_visual_artifacts(
    *,
    intermediates: Tensor,
    time_grid: Tensor,
    tokenizer: PreTrainedTokenizer,
    sample_indices: Iterable[int],
    valid_lengths: Sequence[int],
    output_dir: Path,
    rank: int,
    step: int,
    save_html: bool = True,
    save_json: bool = True,
    color_map: str = "jet",
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    token_order = compute_token_order(intermediates=intermediates)

    for sample_idx in sample_indices:
        sample_prefix = f"visual_rank{rank}_sample{sample_idx}"
        if save_html:
            _ensure_rich_available()
            html_console = Console(record=True, force_terminal=True, width=180)
            render_rich_timeline(
                intermediates=intermediates,
                time_grid=time_grid,
                tokenizer=tokenizer,
                sample_indices=[sample_idx],
                valid_lengths=valid_lengths,
                color_map=color_map,
                console=html_console,
            )
            html_console.save_html(str(output_dir / f"{sample_prefix}.html"))

        if save_json:
            sample_states = intermediates[:, sample_idx]
            changed_counts = [0]
            for i in range(1, sample_states.shape[0]):
                changed_counts.append(int(sample_states[i - 1].ne(sample_states[i]).sum().item()))

            final_len = int(valid_lengths[-1]) if valid_lengths else int(sample_states.shape[1])
            payload = {
                "step": int(step),
                "rank": int(rank),
                "sample_index": int(sample_idx),
                "time_grid": [float(v) for v in time_grid.detach().cpu().tolist()],
                "valid_lengths": [int(v) for v in valid_lengths],
                "final_token_ids": [int(v) for v in sample_states[-1, :final_len].detach().cpu().tolist()],
                "token_update_order": [
                    float(v) for v in token_order[sample_idx, :final_len].detach().cpu().tolist()
                ],
                "color_map": color_map,
                "changed_tokens_per_step": changed_counts,
            }
            with open(output_dir / f"{sample_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

