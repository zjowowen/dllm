from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseKappaScheduler, CubicKappaScheduler
from dllm.pipelines.ctmc_utils import pad_1d
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
        max_w: float = 20.0
        image_loss_weight: float = 1.0
        normalize_text_loss_by_length: bool = True
        normalize_image_loss_by_tokens: bool = True

    def __init__(
        self,
        args: OneFlowConfig,
        scheduler: Optional[BaseKappaScheduler] = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.scheduler = scheduler if scheduler is not None else CubicKappaScheduler()

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
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

        # ---- sample tau_text ~ Unif[0,2], set t_text=min(1,tau_text) ---------------
        tau_text = 2.0 * torch.rand((B, 1), device=device)
        t_text = torch.minimum(tau_text, torch.ones_like(tau_text))  # [B,1] in [0,1]

        k = self.scheduler.kappa(t_text).to(device)  # [B,1]
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

        # ---- build X_t and bag-of-tokens A_j ---------------------------------------
        xt_list: list[list[int]] = []
        bags_list: list[list[list[int]]] = []  # per-sample: list of bags aligned to xt positions
        # For mixed-modal: store per-sample kept image latents and per-image t_img
        kept_images_list: list[list[torch.Tensor]] = []
        kept_timg_list: list[list[float]] = []

        for x1, kb in zip(x1_ids, k.squeeze(1).tolist()):
            b_idx = len(xt_list)
            if not x1:
                raise ValueError("Empty x1_ids is not supported for OneFlow training.")

            # If prompt_len is provided (SFT), do not edit the prompt prefix:
            # force-keep all prompt tokens so no deletions/insertions occur inside.
            pl: int | None = None
            if prompt_len_list is not None:
                pl = int(prompt_len_list[b_idx])
                if pl <= 0 or pl > len(x1):
                    raise ValueError(
                        f"Invalid prompt_len={pl} for sample {b_idx}: x1 length={len(x1)}"
                    )
                # v1 limitation: do not support conditioning on images inside the prompt.
                if image_token_id is not None and any(
                    int(t) == int(image_token_id) for t in x1[:pl]
                ):
                    raise ValueError(
                        f"prompt_len spans over {ONEFLOW_IMAGE_TOKEN} for sample {b_idx}. "
                        "OneFlow v1 does not support conditioning on prompt images. "
                        "Please place image tokens after the prompt, or omit prompt_len."
                    )

            # per-token keep with prob kappa(t); BOS must be kept
            keep = (torch.rand(len(x1), device=device) < float(kb)).tolist()
            keep[0] = True
            if pl is not None:
                for i in range(pl):
                    keep[i] = True
            # If training with images, force-keep the image placeholder tokens so that
            # image insertion/deletion is governed by the interleaved τ_img schedule.
            if image_token_id is not None:
                for idx, tok_id in enumerate(x1):
                    if int(tok_id) == int(image_token_id):
                        keep[idx] = True

            xt: list[int] = []
            bags: list[list[int]] = []
            for token_id, is_keep in zip(x1, keep):
                if is_keep:
                    xt.append(int(token_id))
                    bags.append([])
                else:
                    # BOS is forced kept, so bags is non-empty here
                    bags[-1].append(int(token_id))

            # --- interleaved image schedule (Algorithm 3, lines 16-28) ---
            # If no images, keep as-is.
            if not has_images:
                xt_list.append(xt)
                bags_list.append(bags)
                kept_images_list.append([])
                kept_timg_list.append([])
                continue

            # Standardize images for this sample
            raw = image_latents_raw[b_idx]
            if raw is None:
                images = []
            elif isinstance(raw, list):
                images = raw
            else:
                images = [raw]

            # Validate count: images must match number of image tokens in ground truth x1
            num_img_tokens_x1 = sum(int(t) == int(image_token_id) for t in x1)
            if num_img_tokens_x1 != len(images):
                raise ValueError(
                    f"Mismatch between number of {ONEFLOW_IMAGE_TOKEN} tokens in x1 "
                    f"({num_img_tokens_x1}) and provided image latents ({len(images)})."
                )

            # Locate image tokens in xt (forced kept, so count must match)
            img_pos_xt = [i for i, t in enumerate(xt) if int(t) == int(image_token_id)]
            if len(img_pos_xt) != len(images):
                raise ValueError(
                    f"Internal error: expected {len(images)} image tokens in X_t, got {len(img_pos_xt)}."
                )

            # Sample τ_img per image: τ_img = τ_text - κ^{-1}(u)
            tau_text_b = float(tau_text[b_idx, 0].item())
            delete_flags: list[bool] = []
            timgs: list[float] = []
            for _ in images:
                u = torch.rand((), device=device)
                inv = self.scheduler.kappa_inverse(u)
                tau_img = tau_text_b - float(inv.item() if isinstance(inv, torch.Tensor) else inv)
                if tau_img < 0.0:
                    delete_flags.append(True)
                    timgs.append(0.0)
                else:
                    delete_flags.append(False)
                    timgs.append(float(min(1.0, tau_img)))

            # Remove deleted images from xt/bags (right-to-left to keep indices stable)
            for j in reversed(range(len(images))):
                if not delete_flags[j]:
                    continue
                pos = img_pos_xt[j]
                if pos <= 0:
                    raise AssertionError("Image token cannot be at position 0 (BOS slot).")
                # Add the deleted image token into the previous bag, and merge the bag-after-image.
                bags[pos - 1].append(int(image_token_id))
                bags[pos - 1].extend(bags[pos])
                del bags[pos]
                del xt[pos]

            # Keep remaining images in order
            kept_images: list[torch.Tensor] = []
            kept_timgs: list[float] = []
            for img, is_del, ti in zip(images, delete_flags, timgs):
                if is_del:
                    continue
                kept_images.append(img)
                kept_timgs.append(ti)

            # Validate remaining image token count matches kept images
            img_pos_xt_after = [i for i, t in enumerate(xt) if int(t) == int(image_token_id)]
            if len(img_pos_xt_after) != len(kept_images):
                raise ValueError(
                    f"After τ_img deletion, expected {len(kept_images)} image tokens in X_t "
                    f"but found {len(img_pos_xt_after)}."
                )

            xt_list.append(xt)
            bags_list.append(bags)
            kept_images_list.append(kept_images)
            kept_timg_list.append(kept_timgs)

        pad_id = int(self.processing_class.pad_token_id)

        # If no images are present in the batch, keep the simple text-only path
        if not has_images:
            # ---- pad X_t for the model --------------------------------------------
            x_tok, x_mask = pad_1d(xt_list, pad_val=pad_id)  # [B,L], [B,L]
            x_tok = x_tok.to(device)
            x_mask = x_mask.to(device)

            # per-token time conditioning (broadcast t_text across positions)
            Lmax = x_tok.shape[1]
            times = t_text.expand(B, Lmax)  # [B,L]

            # ---- forward ----------------------------------------------------------
            out = model(
                input_ids=x_tok,
                attention_mask=x_mask,
                is_any_modality=torch.zeros_like(x_mask, dtype=torch.bool),
                modality_tokens=None,
                modality_positions=None,
                times=times,
            )

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

            logQ = F.log_softmax(q_logits, dim=-1)

            def safe_log(x: torch.Tensor) -> torch.Tensor:
                return torch.log(x.clamp_min(1e-12))

            # ---- text survival term -----------------------------------------------
            mask_f = x_mask.float()
            Lambda_hat = (lam * mask_f).sum(dim=1)  # [B]
            L1 = torch.tensor([len(x) for x in x1_ids], device=device, dtype=torch.float)
            denom = (
                L1.clamp_min(1.0)
                if bool(getattr(self.args, "normalize_text_loss_by_length", True))
                else torch.ones_like(L1)
            )
            loss_surv = ((w * Lambda_hat) / denom).mean()

            # ---- positive term -----------------------------------------------------
            pos_terms = []
            for b in range(B):
                lp = x_tok.new_zeros((), dtype=torch.float32)
                cur_len = int(x_mask[b].sum().item())
                for i in range(cur_len):
                    bag = bags_list[b][i]
                    if not bag:
                        continue
                    lp = lp - safe_log(lam[b, i]) * float(len(bag))
                    tok = torch.tensor(bag, device=device, dtype=torch.long)
                    lp = lp - logQ[b, i].gather(dim=-1, index=tok).sum()
                pos_terms.append(lp)
            loss_pos_per = torch.stack(pos_terms)  # [B]
            loss_pos = ((w * loss_pos_per) / denom).mean()

            loss_text = loss_surv + loss_pos
            loss = loss_text + anchor
            return (loss, out) if return_outputs else loss

        # ---- mixed-modal path: build unified sequences with inserted latent tokens ---
        dim_latent = int(getattr(getattr(model, "config", None), "dim_latent", 4))

        def flatten_latent(y: torch.Tensor) -> torch.Tensor:
            # Accept shapes:
            # - [N, d]
            # - [d, H, W] (channel-first)
            # - [H, W, d] (channel-last)
            if y.dim() == 2 and y.shape[-1] == dim_latent:
                return y
            if y.dim() == 3:
                if y.shape[0] == dim_latent:
                    y = y.permute(1, 2, 0).contiguous()  # [H,W,d]
                elif y.shape[-1] != dim_latent:
                    raise ValueError(f"Unrecognized latent shape: {tuple(y.shape)}")
                return y.reshape(-1, dim_latent)
            raise ValueError(f"Unrecognized latent tensor ndim={y.dim()} shape={tuple(y.shape)}")

        total_ids_list: list[list[int]] = []
        total_is_mod_list: list[list[bool]] = []
        total_mod_tokens_list: list[list[torch.Tensor]] = []
        total_flow_targets_list: list[list[torch.Tensor]] = []
        total_times_list: list[list[float]] = []
        modality_positions_list: list[list[tuple[int, int, int]]] = []
        xt_to_total_pos_list: list[list[int]] = []

        zero_lat = torch.zeros((dim_latent,), device=device, dtype=torch.float32)

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
            img_counter = 0

            for i, tok_id in enumerate(xt):
                xt_to_total.append(len(ids))
                ids.append(int(tok_id))
                is_mod.append(False)
                mod_tokens.append(zero_lat)
                flow_targets.append(zero_lat)
                times_b.append(t_text_b)

                if int(tok_id) == int(image_token_id) and img_counter < len(images):
                    y1 = images[img_counter].to(device=device, dtype=torch.float32)
                    ti = float(timgs[img_counter])
                    y0 = torch.randn_like(y1)
                    yt = ti * y1 + (1.0 - ti) * y0
                    flow = y1 - y0

                    yt_tok = flatten_latent(yt)
                    flow_tok = flatten_latent(flow)
                    n_img = int(yt_tok.shape[0])

                    # modality block starts at next position
                    offset = len(ids)
                    mod_pos.append((0, offset, n_img))

                    for j in range(n_img):
                        ids.append(pad_id)  # dummy ids for modality tokens
                        is_mod.append(True)
                        mod_tokens.append(yt_tok[j])
                        flow_targets.append(flow_tok[j])
                        times_b.append(ti)

                    img_counter += 1

            if img_counter != len(images):
                raise ValueError(
                    f"Did not consume all kept images: used {img_counter}, expected {len(images)}."
                )
            if len(bags) != len(xt_to_total):
                raise AssertionError("bags must align with X_t token positions.")

            total_ids_list.append(ids)
            total_is_mod_list.append(is_mod)
            total_mod_tokens_list.append(mod_tokens)
            total_flow_targets_list.append(flow_targets)
            total_times_list.append(times_b)
            modality_positions_list.append(mod_pos)
            xt_to_total_pos_list.append(xt_to_total)

        # pad unified sequences
        x_tok, x_mask = pad_1d(total_ids_list, pad_val=pad_id)  # [B,L], [B,L]
        x_tok = x_tok.to(device)
        x_mask = x_mask.to(device)
        Lmax = x_tok.shape[1]

        # pad modality tokens / flow targets / times / is_modality
        mod_tok = torch.zeros((B, Lmax, dim_latent), device=device, dtype=torch.float32)
        flow_tgt = torch.zeros((B, Lmax, dim_latent), device=device, dtype=torch.float32)
        is_mod = torch.zeros((B, Lmax), device=device, dtype=torch.bool)
        times = torch.zeros((B, Lmax), device=device, dtype=torch.float32)

        for b in range(B):
            Lb = len(total_ids_list[b])
            is_mod[b, :Lb] = torch.tensor(total_is_mod_list[b], device=device, dtype=torch.bool)
            times[b, :Lb] = torch.tensor(total_times_list[b], device=device, dtype=torch.float32)
            mod_tok[b, :Lb] = torch.stack(total_mod_tokens_list[b], dim=0)
            flow_tgt[b, :Lb] = torch.stack(total_flow_targets_list[b], dim=0)

        # pad modality_positions
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

        logQ = F.log_softmax(q_logits, dim=-1)

        def safe_log(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x.clamp_min(1e-12))

        # ---- text survival term (only over X_t token positions, not modality tokens) ---
        Lambda_hat = torch.zeros((B,), device=device, dtype=torch.float32)
        for b in range(B):
            pos_idx = torch.tensor(xt_to_total_pos_list[b], device=device, dtype=torch.long)
            Lambda_hat[b] = lam[b].gather(dim=0, index=pos_idx).sum()

        L1 = torch.tensor([len(x) for x in x1_ids], device=device, dtype=torch.float)
        denom = (
            L1.clamp_min(1.0)
            if bool(getattr(self.args, "normalize_text_loss_by_length", True))
            else torch.ones_like(L1)
        )
        loss_surv = ((w * Lambda_hat) / denom).mean()

        # ---- positive term (bag-of-tokens insertions) ------------------------------
        pos_terms = []
        for b in range(B):
            lp = x_tok.new_zeros((), dtype=torch.float32)
            xt_pos = xt_to_total_pos_list[b]
            for i, pos in enumerate(xt_pos):
                bag = bags_list[b][i]
                if not bag:
                    continue
                # -sum_a (log λ_i + log Q_i(a))
                lp = lp - safe_log(lam[b, pos]) * float(len(bag))
                # token logprobs
                tok = torch.tensor(bag, device=device, dtype=torch.long)
                lp = lp - logQ[b, pos].gather(dim=-1, index=tok).sum()
            pos_terms.append(lp)

        loss_pos_per = torch.stack(pos_terms)  # [B]
        loss_pos = ((w * loss_pos_per) / denom).mean()

        loss_text = loss_surv + loss_pos

        # ---- image flow matching loss --------------------------------------------
        v = out["v"]  # [B,L,dim_latent]
        img_mask = is_mod.float()  # [B,L]
        if img_mask.sum().item() > 0:
            sq = (v - flow_tgt).pow(2).sum(dim=-1)  # [B,L]
            img_sum = (sq * img_mask).sum()
            if bool(getattr(self.args, "normalize_image_loss_by_tokens", True)):
                denom_img = img_mask.sum().clamp_min(1.0)
                loss_img = img_sum / denom_img
            else:
                loss_img = img_sum / float(B)
        else:
            loss_img = torch.zeros((), device=device, dtype=torch.float32)

        image_w = float(getattr(self.args, "image_loss_weight", 1.0))
        loss = loss_text + image_w * loss_img + anchor
        return (loss, out) if return_outputs else loss


