from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseKappaScheduler, LinearKappaScheduler
from dllm.pipelines.ctmc_utils import pad_1d
from dllm.pipelines.oneflow.losses import (
    ctmc_loss_vectorized,
    image_loss_flow_matching,
    safe_log,
    text_loss_paper_eq7_fast,
    text_loss_paper_eq7_fast_from_logits,
)
from dllm.pipelines.oneflow.sequence_ops import (
    apply_interleaved_image_schedule,
    build_noised_xt_and_bags,
    build_unified_train_batch,
    sample_tau_text,
    tau_to_t_text,
)
from dllm.pipelines.oneflow.runtime_config import build_runtime_config, save_runtime_config
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_TOKEN
from dllm.utils.configs import TrainingArguments


class OneFlowTrainer(transformers.Trainer):
    """
    Trainer for OneFlow.

    v1 will implement OneFlow Algorithm 3:
    - sample tau_text, derive t_text
    - build X_t and bag-of-tokens targets A_j
    - interleaved image time tau_img with kappa^{-1}
    - compute L_text + L_image
    """

    @dataclass
    class OneFlowConfig(TrainingArguments):
        time_epsilon: float = 1e-3
        # Upper bound for τ_text sampling (default 2.0 per paper).
        # Set to 1.0 to sample τ_text ~ Unif[0,1] for text-only validation.
        tau_text_max: float = 2.0
        # Paper (Sec 2.1.1): insertion predictions are t-independent in practice.
        # If False, we do NOT condition text-token heads (π/λ/Q) on t_text (times are set constant for text tokens).
        condition_text_on_time: bool = False
        # Text loss type:
        # - "paper": Eq (7) with π BCE + Poisson on λ_nonzero for k>0, no w(t) reweighting.
        # - "ctmc": legacy CTMC-style survival+positive term (EditFlow-like), weighted by w(t).
        text_loss_type: str = "paper"
        # Only used for text_loss_type="ctmc" (legacy): clamp w(t)=κ'(t)/(1-κ(t)) to avoid blow-ups.
        max_w: float = 20.0
        # When True and text_loss_type="paper", multiply the per-sample Eq.7 loss by w(t).
        # This gives: loss = (loss_tok + loss_pi + loss_lam) * w(t).
        # When False (default), Eq.7 loss is unweighted.
        paper_loss_use_w: bool = False
        image_loss_weight: float = 1.0
        normalize_text_loss_by_length: bool = True
        normalize_image_loss_by_tokens: bool = True
        # Debug helpers (off by default)
        debug_log_first_batch: bool = False
        # Log split losses (text/image) to logger integrations (e.g., W&B).
        # These will be emitted at the same cadence as Trainer's `logging_steps`.
        log_split_losses: bool = False
        # Paper loss implementation:
        # - False (default): compute full log-softmax then gather (usually faster / lower peak mem on NPU)
        # - True: compute token CE from logits via logsumexp (may be slower / higher mem on some backends)
        paper_loss_from_logits: bool = False
        # ---- perf profiling helpers (off by default) ------------------------------
        # Log step breakdown timings (ms) at `logging_steps` cadence.
        profile_timing: bool = False
        # Whether to synchronize device between timing sections for more accurate timings.
        # Note: this can add overhead; keep False unless profiling.
        profile_timing_sync: bool = False
        # Log optimizer step time (ms) at `logging_steps` cadence.
        profile_log_optimizer_time: bool = False

    def __init__(
        self,
        args: OneFlowConfig,
        scheduler: Optional[BaseKappaScheduler] = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.scheduler = scheduler if scheduler is not None else LinearKappaScheduler()
        # For split-loss logging at `logging_steps` cadence.
        self._oneflow_pending_logs: dict[int, dict[str, float]] = {}
        self._oneflow_ga_micro_step: int = 0
        self._oneflow_step_text_sum: float = 0.0
        self._oneflow_step_img_sum: float = 0.0
        self._oneflow_step_imgtok_sum: float = 0.0
        self._oneflow_step_count: int = 0
        self._oneflow_step_extra_sums: dict[str, float] = {}
        self._oneflow_interval_text_sum: float = 0.0
        self._oneflow_interval_img_sum: float = 0.0
        self._oneflow_interval_imgtok_sum: float = 0.0
        self._oneflow_interval_count: int = 0
        self._oneflow_interval_extra_sums: dict[str, float] = {}
        # ---- perf timing buffers (rank0) -----------------------------------------
        self._oneflow_last_profile: dict[str, float] | None = None
        self._oneflow_prof_ga_micro_step: int = 0
        self._oneflow_prof_step_sums: dict[str, float] = {}
        self._oneflow_prof_step_count: int = 0
        self._oneflow_prof_interval_sums: dict[str, float] = {}
        self._oneflow_prof_interval_count: int = 0

    @staticmethod
    def _maybe_sync_device(device: torch.device) -> None:
        """
        Best-effort device sync for accurate wall timings when requested.
        """
        try:
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elif device.type == "npu":
                # torch-npu exposes torch.npu in many builds
                if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
                    torch.npu.synchronize()
                else:  # pragma: no cover
                    import torch_npu

                    torch_npu.npu.synchronize()
        except Exception:
            # Never let profiling crash training
            return

    def _oneflow_track_profile_timings(
        self,
        *,
        model,
        timings_ms: dict[str, float],
    ) -> None:
        """
        Track step breakdown timings and schedule them to be logged at `logging_steps`.

        This mirrors `_oneflow_track_split_losses` behavior: we reduce across ranks, average across
        gradient accumulation micro-steps, then average across the logging interval.
        """
        if not bool(getattr(self.args, "profile_timing", False)):
            return
        if model is None or (hasattr(model, "training") and (not model.training)):
            return

        # ---- reduce across ranks -------------------------------------------------
        keys = [
            "time_noising_ms",
            "time_pad_ms",
            "time_forward_ms",
            "time_loss_ms",
            "time_train_step_ms",
        ]
        dev = None
        try:
            dev = next(model.parameters()).device
        except Exception:
            dev = torch.device("cpu")
        vals = torch.tensor([float(timings_ms.get(k, 0.0)) for k in keys], device=dev, dtype=torch.float32)
        try:
            vals = self.accelerator.reduce(vals, reduction="mean")
        except Exception:
            pass

        # Only rank0 maintains logging buffers.
        if not self.is_world_process_zero():
            return

        local = {k: float(vals[i].item()) for i, k in enumerate(keys)}

        # ---- micro-step accumulation (for gradient accumulation) -----------------
        self._oneflow_prof_ga_micro_step += 1
        for k, v in local.items():
            self._oneflow_prof_step_sums[k] = float(self._oneflow_prof_step_sums.get(k, 0.0) + float(v))
        self._oneflow_prof_step_count += 1

        ga = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        ga = max(1, ga)
        if (self._oneflow_prof_ga_micro_step % ga) != 0:
            return

        # finalize one optimizer step average
        denom = max(1, self._oneflow_prof_step_count)
        step_avg = {k: float(v / denom) for k, v in self._oneflow_prof_step_sums.items()}
        self._oneflow_prof_step_sums = {}
        self._oneflow_prof_step_count = 0

        # ---- interval accumulation ----------------------------------------------
        for k, v in step_avg.items():
            self._oneflow_prof_interval_sums[k] = float(self._oneflow_prof_interval_sums.get(k, 0.0) + float(v))
        self._oneflow_prof_interval_count += 1

        logging_steps = int(getattr(self.args, "logging_steps", 0) or 0)
        logging_steps = max(1, logging_steps)
        next_step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1
        if (next_step % logging_steps) != 0:
            return

        denom2 = max(1, self._oneflow_prof_interval_count)
        to_log = {k: float(v / denom2) for k, v in self._oneflow_prof_interval_sums.items()}
        # Convert to shorter keys in logs.
        payload = {
            "time_noising_ms": float(to_log.get("time_noising_ms", 0.0)),
            "time_pad_ms": float(to_log.get("time_pad_ms", 0.0)),
            "time_forward_ms": float(to_log.get("time_forward_ms", 0.0)),
            "time_loss_ms": float(to_log.get("time_loss_ms", 0.0)),
            "time_train_step_ms": float(to_log.get("time_train_step_ms", 0.0)),
        }
        self._oneflow_pending_logs[int(next_step)] = {
            **self._oneflow_pending_logs.get(int(next_step), {}),
            **payload,
        }

        self._oneflow_prof_interval_sums = {}
        self._oneflow_prof_interval_count = 0

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        """
        Wrap HF Trainer training_step to collect a coarse-grained train-step wall time.
        """
        prof_on = bool(getattr(self.args, "profile_timing", False))
        if not prof_on:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        dev = None
        try:
            dev = next(model.parameters()).device
        except Exception:
            dev = torch.device("cpu")

        if bool(getattr(self.args, "profile_timing_sync", False)):
            self._maybe_sync_device(dev)
        t0 = time.perf_counter()
        out = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if bool(getattr(self.args, "profile_timing_sync", False)):
            self._maybe_sync_device(dev)
        t1 = time.perf_counter()

        # Merge with compute_loss breakdown (captured in compute_loss).
        breakdown = self._oneflow_last_profile or {}
        breakdown = dict(breakdown)
        breakdown["time_train_step_ms"] = (t1 - t0) * 1000.0
        self._oneflow_last_profile = None
        self._oneflow_track_profile_timings(model=model, timings_ms=breakdown)
        return out

    def optimizer_step(self, *args, **kwargs):  # type: ignore[override]
        """
        Optionally time optimizer step and schedule it to be logged at `logging_steps` cadence.
        """
        if not bool(getattr(self.args, "profile_log_optimizer_time", False)):
            return super().optimizer_step(*args, **kwargs)

        model = getattr(self, "model", None)
        dev = None
        try:
            dev = next(model.parameters()).device if model is not None else torch.device("cpu")
        except Exception:
            dev = torch.device("cpu")

        if bool(getattr(self.args, "profile_timing_sync", False)):
            self._maybe_sync_device(dev)
        t0 = time.perf_counter()
        out = super().optimizer_step(*args, **kwargs)
        if bool(getattr(self.args, "profile_timing_sync", False)):
            self._maybe_sync_device(dev)
        t1 = time.perf_counter()

        # Reduce across ranks, then schedule for logging on rank0.
        ms = torch.tensor([(t1 - t0) * 1000.0], device=dev, dtype=torch.float32)
        try:
            ms = self.accelerator.reduce(ms, reduction="mean")
        except Exception:
            pass

        if self.is_world_process_zero():
            logging_steps = int(getattr(self.args, "logging_steps", 0) or 0)
            logging_steps = max(1, logging_steps)
            next_step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1
            if (next_step % logging_steps) == 0:
                self._oneflow_pending_logs[int(next_step)] = {
                    **self._oneflow_pending_logs.get(int(next_step), {}),
                    "time_optim_step_ms": float(ms.item()),
                }
        return out

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Inject pending OneFlow-specific metrics (e.g., split losses) into Trainer logs.

        Trainer's internal logging uses `self.state.global_step` after optimizer step.
        We queue metrics keyed by that step in `self._oneflow_pending_logs`, then
        merge them here right before callbacks (W&B/TensorBoard) consume the logs.
        """
        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        extra = self._oneflow_pending_logs.pop(step, None) if isinstance(self._oneflow_pending_logs, dict) else None
        if extra:
            merged = dict(logs)
            merged.update(extra)
            return super().log(merged, *args, **kwargs)
        return super().log(logs, *args, **kwargs)

    def _oneflow_track_split_losses(
        self,
        *,
        model,
        loss_text: torch.Tensor,
        loss_img: torch.Tensor,
        img_tokens_total: float,
        extra_metrics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Track split losses and schedule them to be logged at `logging_steps`.

        Notes:
        - We reduce losses across distributed ranks (mean) so the curve is stable.
        - We average across gradient accumulation steps to match optimizer-step semantics.
        - We further average across the interval between logs, mirroring Trainer's `loss`.
        """
        if not bool(getattr(self.args, "log_split_losses", False)):
            return
        if model is None or (hasattr(model, "training") and (not model.training)):
            return

        # Reduce across ranks (collective); must run on all processes.
        lt = loss_text.detach()
        li = loss_img.detach()
        it = torch.tensor(float(img_tokens_total), device=lt.device, dtype=torch.float32)
        extra_metrics = extra_metrics or {}
        extra_reduced: dict[str, torch.Tensor] = {}
        try:
            lt = self.accelerator.reduce(lt, reduction="mean")
            li = self.accelerator.reduce(li, reduction="mean")
            it = self.accelerator.reduce(it, reduction="mean")
            for name, val in extra_metrics.items():
                extra_reduced[name] = self.accelerator.reduce(val.detach(), reduction="mean")
        except Exception:
            # Single-process / no accelerator reduce available.
            for name, val in extra_metrics.items():
                extra_reduced[name] = val.detach()

        # Only rank0 maintains logging buffers (after collectives have run).
        if not self.is_world_process_zero():
            return

        # ---- micro-step accumulation (for gradient accumulation) -----------------
        self._oneflow_ga_micro_step += 1
        self._oneflow_step_text_sum += float(lt.item())
        self._oneflow_step_img_sum += float(li.item())
        self._oneflow_step_imgtok_sum += float(it.item())
        for name, val in extra_reduced.items():
            self._oneflow_step_extra_sums[name] = self._oneflow_step_extra_sums.get(name, 0.0) + float(
                val.item()
            )
        self._oneflow_step_count += 1

        ga = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        ga = max(1, ga)
        if (self._oneflow_ga_micro_step % ga) != 0:
            return

        # finalize one optimizer step average
        step_text = self._oneflow_step_text_sum / max(1, self._oneflow_step_count)
        step_img = self._oneflow_step_img_sum / max(1, self._oneflow_step_count)
        step_imgtok = self._oneflow_step_imgtok_sum / max(1, self._oneflow_step_count)
        step_extra = {
            name: val / max(1, self._oneflow_step_count)
            for name, val in self._oneflow_step_extra_sums.items()
        }

        self._oneflow_step_text_sum = 0.0
        self._oneflow_step_img_sum = 0.0
        self._oneflow_step_imgtok_sum = 0.0
        self._oneflow_step_extra_sums = {}
        self._oneflow_step_count = 0

        # ---- interval accumulation (to mirror Trainer's `loss`) -------------------
        self._oneflow_interval_text_sum += float(step_text)
        self._oneflow_interval_img_sum += float(step_img)
        self._oneflow_interval_imgtok_sum += float(step_imgtok)
        for name, val in step_extra.items():
            self._oneflow_interval_extra_sums[name] = (
                self._oneflow_interval_extra_sums.get(name, 0.0) + float(val)
            )
        self._oneflow_interval_count += 1

        logging_steps = int(getattr(self.args, "logging_steps", 0) or 0)
        logging_steps = max(1, logging_steps)
        next_step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1
        if (next_step % logging_steps) != 0:
            return

        denom = max(1, self._oneflow_interval_count)
        extra_logs = {
            name: float(val / denom) for name, val in self._oneflow_interval_extra_sums.items()
        }
        self._oneflow_pending_logs[int(next_step)] = {
            "loss_text": float(self._oneflow_interval_text_sum / denom),
            "loss_img": float(self._oneflow_interval_img_sum / denom),
            "img_tokens_total": float(self._oneflow_interval_imgtok_sum / denom),
            **extra_logs,
        }

        # reset interval buffers after scheduling a log
        self._oneflow_interval_text_sum = 0.0
        self._oneflow_interval_img_sum = 0.0
        self._oneflow_interval_imgtok_sum = 0.0
        self._oneflow_interval_count = 0
        self._oneflow_interval_extra_sums = {}

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Ensure intermediate Trainer checkpoints are loadable by OneFlow utilities.

        HF Trainer saves weights as `model.safetensors` by default, and (since our model
        is not a HF PreTrainedModel) it won't save our custom `oneflow_config.json`.

        We keep Trainer's default behavior (weights/tokenizer/training_args), and also
        write:
        - `oneflow_config.json` (model architecture config)
        - `oneflow_runtime_config.json` (train/eval/sample consistency metadata)
        """
        super()._save(output_dir=output_dir, state_dict=state_dict)

        out_dir = output_dir if output_dir is not None else self.args.output_dir
        cfg_path = os.path.join(out_dir, "oneflow_config.json")
        if not os.path.exists(cfg_path):
            # Unwrap model from DDP/Accelerate wrappers.
            try:
                unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            except Exception:
                unwrapped = getattr(self.model, "module", self.model)

            cfg = getattr(unwrapped, "config", None)
            cfg_dict = None
            if dataclasses.is_dataclass(cfg):
                cfg_dict = dataclasses.asdict(cfg)
            elif hasattr(cfg, "to_dict"):
                cfg_dict = cfg.to_dict()
            elif isinstance(cfg, dict):
                cfg_dict = cfg

            if cfg_dict is not None:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

        runtime_cfg = build_runtime_config(
            training_args=self.args,
            scheduler=self.scheduler,
        )
        save_runtime_config(out_dir, runtime_cfg)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        prof_on = bool(getattr(self.args, "profile_timing", False))
        prof_sync = bool(getattr(self.args, "profile_timing_sync", False))
        prof_times: dict[str, float] = {}

        # ---- inputs (text-only v1) -------------------------------------------------
        if "x1_ids" not in inputs:
            raise KeyError(
                "OneFlowTrainer expects `x1_ids` in the batch. "
                "Did you forget to use OneFlowCollator?"
            )

        x1_ids: list[list[int]] = inputs["x1_ids"]
        B = len(x1_ids)

        prompt_len_raw = inputs.get("prompt_len", None)
        prompt_len_list: list[int] | None = None
        if prompt_len_raw is not None:
            if isinstance(prompt_len_raw, torch.Tensor):
                pl = prompt_len_raw.detach().to("cpu")
                if pl.ndim == 2 and pl.shape[1] == 1:
                    pl = pl.squeeze(1)
                prompt_len_list = [int(x) for x in pl.tolist()]
            else:
                prompt_len_list = [int(x) for x in prompt_len_raw]
            if len(prompt_len_list) != B:
                raise ValueError(
                    f"prompt_len batch size mismatch: got {len(prompt_len_list)} values, expected {B}."
                )

        device = next(model.parameters()).device

        if prof_on and prof_sync:
            self._maybe_sync_device(device)

        # ---- sample τ_text and derive t_text --------------------------------------
        # IMPORTANT (perf): discrete noising builds python lists (X_t + bags) and historically
        # used `.tolist()` on NPU tensors, which forces NPU->CPU synchronization. For text-only,
        # we keep τ/t/κ and keep-mask sampling on CPU, then only move padded tensors to NPU.
        t0 = time.perf_counter() if prof_on else 0.0
        cpu = torch.device("cpu")
        tau_text_cpu = sample_tau_text(
            batch_size=B,
            device=cpu,
            tau_text_max=float(getattr(self.args, "tau_text_max", 2.0) or 2.0),
        )
        t_text_cpu = tau_to_t_text(tau_text_cpu)  # [B,1] in [0,1]

        # Keep a device copy for model time conditioning and (optional) mixed-modal schedule.
        tau_text = tau_text_cpu.to(device)
        t_text = t_text_cpu.to(device)

        # Token keep prob κ(t_text) for discrete noising (compute on CPU to avoid device sync)
        k_keep_cpu = self.scheduler.kappa(t_text_cpu).to(cpu)  # [B,1]

        # Legacy CTMC-style loss uses w(t)=κ'(t)/(1-κ(t)); paper loss (Eq 7) does NOT by default.
        # When paper_loss_use_w=True, paper loss is also weighted by w(t).
        text_loss_type = str(getattr(self.args, "text_loss_type", "paper") or "paper").lower().strip()
        paper_loss_use_w = bool(getattr(self.args, "paper_loss_use_w", False))
        w: torch.Tensor | None = None
        if text_loss_type == "ctmc" or paper_loss_use_w:
            w = self.scheduler.weight(t_text).squeeze(1).to(device)  # [B]
            if getattr(self.args, "max_w", None):
                w = w.clamp(max=float(self.args.max_w))

        # ---- optional images (pre-encoded latents) --------------------------------
        # `image_latents[b]` can be:
        # - None (text-only)
        # - a Tensor for single image
        # - a list[Tensor] for multiple images
        image_latents_raw = inputs.get("image_latents", None)
        has_images = image_latents_raw is not None and any(
            x is not None for x in image_latents_raw
        )

        # resolve image token id if needed
        image_token_id: int | None = None
        if has_images:
            tok = self.processing_class
            image_token_id = int(tok.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))
            # If token is unknown, treat as not configured.
            if tok.unk_token_id is not None and image_token_id == int(tok.unk_token_id):
                raise ValueError(
                    f"Tokenizer does not recognize {ONEFLOW_IMAGE_TOKEN}. "
                    "Please add it as a special token before training."
                )

        # ---- build X_t and bag-of-tokens A_i --------------------------------------
        noised = build_noised_xt_and_bags(
            x1_ids=x1_ids,
            kappa_keep=k_keep_cpu,
            device=cpu,
            prompt_len_list=prompt_len_list,
            image_token_id=image_token_id,
            disallow_image_in_prompt=True,
        )
        xt_list = noised.xt_list
        bags_list = noised.bags_list

        if prof_on:
            if prof_sync:
                self._maybe_sync_device(device)
            prof_times["time_noising_ms"] = (time.perf_counter() - t0) * 1000.0

        kept_images_list: list[list[torch.Tensor]] = [[] for _ in range(B)]
        kept_timg_list: list[list[float]] = [[] for _ in range(B)]
        if has_images:
            if image_token_id is None:
                raise RuntimeError("has_images=True but image_token_id is None.")
            inter = apply_interleaved_image_schedule(
                x1_ids=x1_ids,
                xt_list=xt_list,
                bags_list=bags_list,
                image_latents_raw=image_latents_raw,
                tau_text=tau_text,
                scheduler=self.scheduler,
                image_token_id=int(image_token_id),
                device=device,
            )
            xt_list = inter.xt_list
            bags_list = inter.bags_list
            kept_images_list = inter.kept_images_list
            kept_timg_list = inter.kept_timg_list

        pad_id = int(self.processing_class.pad_token_id)

        # If no images are present in the batch, keep the simple text-only path
        if not has_images:
            t_pad0 = time.perf_counter() if prof_on else 0.0
            # ---- pad X_t for the model --------------------------------------------
            x_tok, x_mask = pad_1d(xt_list, pad_val=pad_id)  # [B,L], [B,L]
            x_tok = x_tok.to(device)
            x_mask = x_mask.to(device)

            # per-token time conditioning
            # Paper (Sec 2.1.1) uses t-independent insertions in practice. We follow that by
            # default: text tokens get a constant time value, while the noising schedule is
            # still governed by t_text in κ(t_text).
            Lmax = x_tok.shape[1]
            if bool(getattr(self.args, "condition_text_on_time", False)):
                times = t_text.expand(B, Lmax)  # [B,L]
            else:
                times = torch.zeros((B, Lmax), device=device, dtype=torch.float32)

            if prof_on:
                if prof_sync:
                    self._maybe_sync_device(device)
                prof_times["time_pad_ms"] = (time.perf_counter() - t_pad0) * 1000.0

            # ---- forward ----------------------------------------------------------
            t_fwd0 = time.perf_counter() if prof_on else 0.0
            out = model(
                input_ids=x_tok,
                attention_mask=x_mask,
                is_any_modality=torch.zeros_like(x_mask, dtype=torch.bool),
                modality_tokens=None,
                modality_positions=None,
                times=times,
            )

            if prof_on:
                if prof_sync:
                    self._maybe_sync_device(device)
                prof_times["time_forward_ms"] = (time.perf_counter() - t_fwd0) * 1000.0

            pi = out["pi"]  # [B,L]
            lam = out["lambda_nonzero"]  # [B,L]
            q_logits = out["q_logits"]  # [B,L,V]

            # touch all heads to avoid unused-parameter issues under ZeRO
            anchor = (
                pi.sum() * 0.0
                + lam.sum() * 0.0
                + q_logits.sum() * 0.0
                + out["v"].sum() * 0.0
            )

            # ---- text loss ---------------------------------------------------------
            t_loss0 = time.perf_counter() if prof_on else 0.0
            extra_metrics: dict[str, torch.Tensor] = {}
            if text_loss_type == "ctmc":
                if w is None:
                    raise RuntimeError("text_loss_type='ctmc' requires w(t) but w is None.")
                logQ = F.log_softmax(q_logits, dim=-1)
                # Vectorized CTMC-style loss: survival + positive term, weighted by w(t).
                ctmc = ctmc_loss_vectorized(
                    lam=lam,
                    logQ=logQ,
                    bags_list=bags_list,
                    w=w,
                    x1_lengths=[len(x) for x in x1_ids],
                    xt_positions=None,
                    normalize_by_length=bool(
                        getattr(self.args, "normalize_text_loss_by_length", True)
                    ),
                )
                loss_text = ctmc.total
                extra_metrics = {
                    "loss_text_surv": ctmc.loss_surv,
                    "loss_text_pos": ctmc.loss_pos,
                }
            else:
                paper_w = w if paper_loss_use_w else None
                if bool(getattr(self.args, "paper_loss_from_logits", False)):
                    tl = text_loss_paper_eq7_fast_from_logits(
                        pi=pi,
                        lam=lam,
                        q_logits=q_logits,
                        bags_list=bags_list,
                        xt_positions=None,
                        normalize_by_n=bool(
                            getattr(self.args, "normalize_text_loss_by_length", True)
                        ),
                        w=paper_w,
                    )
                else:
                    logQ = F.log_softmax(q_logits, dim=-1)
                    tl = text_loss_paper_eq7_fast(
                        pi=pi,
                        lam=lam,
                        logQ=logQ,
                        bags_list=bags_list,
                        xt_positions=None,
                        normalize_by_n=bool(
                            getattr(self.args, "normalize_text_loss_by_length", True)
                        ),
                        w=paper_w,
                    )
                loss_text = tl.total
                extra_metrics = {
                    "loss_text_pi": tl.loss_pi,
                    "loss_text_lam": tl.loss_lam,
                    "loss_text_tok": tl.loss_tok,
                }
            if prof_on:
                if prof_sync:
                    self._maybe_sync_device(device)
                prof_times["time_loss_ms"] = (time.perf_counter() - t_loss0) * 1000.0
            loss = loss_text + anchor
            # split-loss logging (text-only: loss_img=0)
            self._oneflow_track_split_losses(
                model=model,
                loss_text=loss_text,
                loss_img=torch.zeros_like(loss_text),
                img_tokens_total=0.0,
                extra_metrics=extra_metrics,
            )
            # ---- optional debug (print once, rank0) ----------------------------------
            if bool(getattr(self.args, "debug_log_first_batch", False)) and self.is_world_process_zero():
                if not hasattr(self, "_oneflow_debug_first_batch_printed"):
                    setattr(self, "_oneflow_debug_first_batch_printed", True)
                    try:
                        step = int(getattr(getattr(self, "state", None), "global_step", -1))
                    except Exception:
                        step = -1
                    x0 = x1_ids[0] if x1_ids else []
                    tok = getattr(self, "processing_class", None)
                    unk_id = int(tok.unk_token_id) if tok is not None and tok.unk_token_id is not None else None
                    num_unk = (
                        int(sum(int(t) == int(unk_id) for t in x0)) if (unk_id is not None and x0) else 0
                    )
                    unk_ratio = (float(num_unk) / float(len(x0))) if x0 else 0.0
                    try:
                        preview = (
                            tok.decode(x0[:80], skip_special_tokens=False)
                            if tok is not None and x0
                            else ""
                        )
                    except Exception:
                        preview = ""
                    # Prefer paper-loss splits when available; else log CTMC splits.
                    if "loss_text_pi" in extra_metrics:
                        pi_v = float(extra_metrics["loss_text_pi"].item())
                        lam_v = float(extra_metrics["loss_text_lam"].item())
                        tok_v = float(extra_metrics["loss_text_tok"].item())
                        loss_detail = f"loss_text_pi={pi_v:.6f} loss_text_lam={lam_v:.6f} loss_text_tok={tok_v:.6f}"
                    elif "loss_text_surv" in extra_metrics:
                        surv_v = float(extra_metrics["loss_text_surv"].item())
                        pos_v = float(extra_metrics["loss_text_pos"].item())
                        loss_detail = f"loss_text_surv={surv_v:.6f} loss_text_pos={pos_v:.6f}"
                    else:
                        loss_detail = ""
                    print(
                        "\n[oneflow-debug] first batch summary:\n"
                        f"  step={step} B={B} has_images={has_images}\n"
                        f"  loss_text={float(loss_text.item()):.6f} {loss_detail}\n"
                        f"  sample0_len={len(x0)} sample0_num_unk={num_unk} "
                        f"sample0_unk_ratio={unk_ratio:.4f}\n"
                        + (f"  sample0_preview={preview}\n" if preview else "")
                    )
            # Store breakdown for `training_step` to aggregate & log.
            if prof_on:
                self._oneflow_last_profile = prof_times
            return (loss, out) if return_outputs else loss

        # ---- mixed-modal path: build unified sequences with inserted latent tokens ---
        dim_latent = int(getattr(getattr(model, "config", None), "dim_latent", 4))
        if image_token_id is None:
            raise RuntimeError("Mixed-modal path requires image_token_id but got None.")

        unified, flow_tgt = build_unified_train_batch(
            xt_list=xt_list,
            bags_list=bags_list,
            kept_images_list=kept_images_list,
            kept_timg_list=kept_timg_list,
            t_text=t_text,
            image_token_id=int(image_token_id),
            pad_id=pad_id,
            dim_latent=dim_latent,
            condition_text_on_time=bool(getattr(self.args, "condition_text_on_time", False)),
            device=device,
        )

        x_tok = unified.input_ids
        x_mask = unified.attention_mask
        is_mod = unified.is_any_modality
        mod_tok = unified.modality_tokens
        flow_tgt = unified.flow_targets
        times = unified.times
        mod_pos_tensor = unified.modality_positions
        xt_to_total_pos_list = unified.xt_to_total_pos_list

        # ---- forward on unified sequence -----------------------------------------
        out = model(
            input_ids=x_tok,
            attention_mask=x_mask,
            is_any_modality=is_mod,
            modality_tokens=mod_tok,
            modality_positions=mod_pos_tensor,
            times=times,
        )

        pi = out["pi"]
        lam = out["lambda_nonzero"]
        q_logits = out["q_logits"]

        # touch all heads to avoid unused-parameter issues under ZeRO
        anchor = (
            pi.sum() * 0.0
            + lam.sum() * 0.0
            + q_logits.sum() * 0.0
            + out["v"].sum() * 0.0
        )

        # ---- text loss (only over X_t token positions, not modality tokens) --------
        extra_metrics: dict[str, torch.Tensor] = {}
        if text_loss_type == "ctmc":
            if w is None:
                raise RuntimeError("text_loss_type='ctmc' requires w(t) but w is None.")
            logQ = F.log_softmax(q_logits, dim=-1)
            # Vectorized CTMC-style loss with xt_positions for mixed-modal.
            ctmc = ctmc_loss_vectorized(
                lam=lam,
                logQ=logQ,
                bags_list=bags_list,
                w=w,
                x1_lengths=[len(x) for x in x1_ids],
                xt_positions=xt_to_total_pos_list,
                normalize_by_length=bool(
                    getattr(self.args, "normalize_text_loss_by_length", True)
                ),
            )
            loss_text = ctmc.total
            extra_metrics = {
                "loss_text_surv": ctmc.loss_surv,
                "loss_text_pos": ctmc.loss_pos,
            }
        else:
            if bool(getattr(self.args, "paper_loss_from_logits", False)):
                tl = text_loss_paper_eq7_fast_from_logits(
                    pi=pi,
                    lam=lam,
                    q_logits=q_logits,
                    bags_list=bags_list,
                    xt_positions=xt_to_total_pos_list,
                    normalize_by_n=bool(getattr(self.args, "normalize_text_loss_by_length", True)),
                )
            else:
                logQ = F.log_softmax(q_logits, dim=-1)
                tl = text_loss_paper_eq7_fast(
                    pi=pi,
                    lam=lam,
                    logQ=logQ,
                    bags_list=bags_list,
                    xt_positions=xt_to_total_pos_list,
                    normalize_by_n=bool(getattr(self.args, "normalize_text_loss_by_length", True)),
                )
            loss_text = tl.total
            extra_metrics = {
                "loss_text_pi": tl.loss_pi,
                "loss_text_lam": tl.loss_lam,
                "loss_text_tok": tl.loss_tok,
            }

        # ---- image flow matching loss --------------------------------------------
        v = out["v"]  # [B,L,dim_latent]
        img = image_loss_flow_matching(
            v=v,
            flow_tgt=flow_tgt,
            is_any_modality=is_mod,
            normalize_by_tokens=bool(getattr(self.args, "normalize_image_loss_by_tokens", True)),
        )
        loss_img = img.loss

        image_w = float(getattr(self.args, "image_loss_weight", 1.0))
        loss = loss_text + image_w * loss_img + anchor
        # split-loss logging (schedule at logging_steps cadence)
        self._oneflow_track_split_losses(
            model=model,
            loss_text=loss_text,
            loss_img=loss_img,
            img_tokens_total=float(img.tokens_total.item()),
            extra_metrics=extra_metrics,
        )

        # ---- optional debug (print once, rank0) ----------------------------------
        if bool(getattr(self.args, "debug_log_first_batch", False)) and self.is_world_process_zero():
            if not hasattr(self, "_oneflow_debug_first_batch_printed"):
                setattr(self, "_oneflow_debug_first_batch_printed", True)
                try:
                    step = int(getattr(getattr(self, "state", None), "global_step", -1))
                except Exception:
                    step = -1
                # show a tiny preview of the first sample for sanity
                x0 = x1_ids[0] if x1_ids else []
                num_img_tok_x0 = (
                    int(sum(int(t) == int(image_token_id) for t in x0)) if image_token_id is not None else 0
                )
                try:
                    preview = (
                        self.processing_class.decode(x0[:80], skip_special_tokens=False)
                        if getattr(self, "processing_class", None) is not None
                        else ""
                    )
                except Exception:
                    preview = ""
                print(
                    "\n[oneflow-debug] first batch summary:\n"
                    f"  step={step} B={B} has_images={has_images}\n"
                    f"  img_tokens_total={float(img.tokens_total.item())} (modality token positions)\n"
                    f"  loss_text={float(loss_text.item()):.6f} loss_img={float(loss_img.item()):.6f} image_w={image_w}\n"
                    f"  sample0_len={len(x0)} sample0_num_image_tokens={num_img_tok_x0}\n"
                    + (f"  sample0_preview={preview}\n" if preview else "")
                )

        return (loss, out) if return_outputs else loss


