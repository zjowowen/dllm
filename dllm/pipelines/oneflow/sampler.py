from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
from dllm.core.schedulers import BaseKappaScheduler, LinearKappaScheduler
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_TOKEN


@dataclass
class OneFlowSamplerConfig(SamplerConfig):
    dt: float = 0.05
    time_epsilon: float = 1e-3
    max_steps: int = 512
    temperature: float = 0.0
    use_pi_gate: bool = True
    image_num_tokens: int = 64  # fixed number of latent tokens per image (v1)


@dataclass
class OneFlowSamplerOutput(SamplerOutput):
    images: list[torch.Tensor] | None = None
    image_times: list[float] | None = None


class OneFlowSampler(BaseSampler):
    """
    Interleaved text-image sampler (Algorithm 1-2).

    v1 will support bs=1 initially (same as EditFlowSampler), then generalize to bs>1.
    """

    kappa_scheduler: BaseKappaScheduler | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.kappa_scheduler is None:
            self.kappa_scheduler = LinearKappaScheduler()

    @torch.no_grad()
    def sample(
        self,
        inputs: List[torch.Tensor | list],
        config: Optional[OneFlowSamplerConfig] = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        if config is None:
            config = OneFlowSamplerConfig()

        dt = float(kwargs.get("dt", config.dt))
        time_epsilon = float(kwargs.get("time_epsilon", config.time_epsilon))
        max_steps = int(kwargs.get("max_steps", config.max_steps))
        temperature = float(kwargs.get("temperature", config.temperature))
        use_pi_gate = bool(kwargs.get("use_pi_gate", config.use_pi_gate))
        image_num_tokens = int(kwargs.get("image_num_tokens", config.image_num_tokens))
        return_dict = bool(kwargs.get("return_dict", config.return_dict))

        if len(inputs) != 1:
            raise NotImplementedError("OneFlowSampler v1 only supports bs=1")

        x0 = inputs[0]
        if isinstance(x0, list):
            x_list = [int(t) for t in x0]
        else:
            x0 = x0.detach().to("cpu")
            if x0.dim() == 2:
                if x0.size(0) != 1:
                    raise NotImplementedError("OneFlowSampler v1 only supports bs=1")
                x0 = x0.squeeze(0)
            x_list = [int(t) for t in x0.tolist()]

        bos = self.tokenizer.bos_token_id
        if bos is None:
            raise ValueError("tokenizer.bos_token_id must be set")
        if len(x_list) == 0:
            x_list = [int(bos)]
        elif x_list[0] != int(bos):
            x_list = [int(bos)] + x_list

        image_token_id = int(self.tokenizer.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))
        if self.tokenizer.unk_token_id is not None and image_token_id == int(
            self.tokenizer.unk_token_id
        ):
            raise ValueError(
                f"Tokenizer does not recognize {ONEFLOW_IMAGE_TOKEN}. "
                "Please add it as a special token before sampling."
            )

        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        device = next(self.model.parameters()).device

        dim_latent = int(getattr(getattr(self.model, "config", None), "dim_latent", 4))
        zero_lat = torch.zeros((dim_latent,), device=device, dtype=torch.float32)

        # images are stored aligned to the order of `<|oneflow_image|>` tokens in x_list
        images: list[dict[str, Any]] = []

        def ensure_images_for_prompt():
            nonlocal images
            n_tokens = sum(1 for t in x_list if int(t) == int(image_token_id))
            while len(images) < n_tokens:
                images.append(
                    dict(
                        latent=torch.randn(
                            (image_num_tokens, dim_latent),
                            device=device,
                            dtype=torch.float32,
                        ),
                        t=0.0,
                    )
                )

        ensure_images_for_prompt()

        def sample_from_logits(logits_row: torch.Tensor) -> int:
            if temperature <= 0.0:
                return int(torch.argmax(logits_row).item())
            return int(
                torch.distributions.Categorical(logits=(logits_row / temperature))
                .sample()
                .item()
            )

        # histories (text-only for v1)
        histories = [] if return_dict else None

        t_text = 0.0
        for _step in range(max_steps):
            ensure_images_for_prompt()

            # ---- build unified sequence (text + modality tokens) ------------------
            ids: list[int] = []
            is_mod: list[bool] = []
            mod_tokens: list[torch.Tensor] = []
            times: list[float] = []

            text_pos_total: list[int] = []
            image_slices: list[tuple[int, int]] = []
            modality_positions: list[tuple[int, int, int]] = []

            img_counter = 0
            for tok in x_list:
                text_pos_total.append(len(ids))
                ids.append(int(tok))
                is_mod.append(False)
                mod_tokens.append(zero_lat)
                times.append(float(t_text))

                if int(tok) == int(image_token_id):
                    if img_counter >= len(images):
                        images.append(
                            dict(
                                latent=torch.randn(
                                    (image_num_tokens, dim_latent),
                                    device=device,
                                    dtype=torch.float32,
                                ),
                                t=0.0,
                            )
                        )
                    y = images[img_counter]["latent"]
                    ti = float(images[img_counter]["t"])
                    n_img = int(y.shape[0])

                    start = len(ids)
                    modality_positions.append((0, start, n_img))

                    for j in range(n_img):
                        ids.append(pad_id)  # dummy id
                        is_mod.append(True)
                        mod_tokens.append(y[j])
                        times.append(ti)

                    end = len(ids)
                    image_slices.append((start, end))
                    img_counter += 1

            # tensors (bs=1, no padding)
            input_ids = torch.tensor([ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            is_any_modality = torch.tensor([is_mod], device=device, dtype=torch.bool)
            modality_tokens = torch.stack(mod_tokens, dim=0).unsqueeze(0)  # [1,N,d]
            times_tensor = torch.tensor([times], device=device, dtype=torch.float32)
            if modality_positions:
                mod_pos = torch.tensor(
                    [modality_positions], device=device, dtype=torch.long
                )  # [1,M,3]
            else:
                mod_pos = None

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_any_modality=is_any_modality,
                modality_tokens=modality_tokens,
                modality_positions=mod_pos,
                times=times_tensor,
            )

            pi = out["pi"][0]  # [N]
            lam = out["lambda_nonzero"][0]  # [N]
            q_logits = out["q_logits"][0]  # [N,V]
            v = out["v"][0]  # [N,dim_latent]

            # ---- image updates (Euler) -------------------------------------------
            for img_idx, (start, end) in enumerate(image_slices):
                t_img = float(images[img_idx]["t"])
                if t_img >= 1.0 - time_epsilon:
                    continue
                dt_img = min(dt, 1.0 - t_img)
                images[img_idx]["latent"] = images[img_idx]["latent"] + dt_img * v[
                    start:end
                ]
                images[img_idx]["t"] = t_img + dt_img

            # ---- text insertions (parallel) --------------------------------------
            dt_text = min(dt, 1.0 - t_text)
            if dt_text > 0.0:
                t_tensor = torch.tensor([[t_text]], device=device, dtype=torch.float32)
                w = float(self.kappa_scheduler.weight(t_tensor).item())

                insertions: list[tuple[int, int]] = []  # (slot_index_in_x, token_id)
                for i, pos in enumerate(text_pos_total):
                    p_lam = (dt_text * w * float(lam[pos].item()))
                    p_lam = max(0.0, min(1.0 - 1e-6, p_lam))
                    do_lam = bool(torch.bernoulli(torch.tensor(p_lam, device=device)).item())
                    if not do_lam:
                        continue

                    if use_pi_gate:
                        p_pi = float(1.0 - pi[pos].item())
                        p_pi = max(0.0, min(1.0 - 1e-6, p_pi))
                        do_pi = bool(torch.bernoulli(torch.tensor(p_pi, device=device)).item())
                        if not do_pi:
                            continue

                    a = sample_from_logits(q_logits[pos])
                    insertions.append((i, a))

                # apply from right to left so indices stay valid
                for i, a in sorted(insertions, key=lambda x: x[0], reverse=True):
                    insert_at = i + 1
                    x_list.insert(insert_at, int(a))

                    # If inserting an image token, also create a new latent and align it in images list.
                    if int(a) == int(image_token_id):
                        img_insert_index = sum(
                            1 for t in x_list[:insert_at] if int(t) == int(image_token_id)
                        )
                        images.insert(
                            img_insert_index,
                            dict(
                                latent=torch.randn(
                                    (image_num_tokens, dim_latent),
                                    device=device,
                                    dtype=torch.float32,
                                ),
                                t=0.0,
                            ),
                        )

                t_text = min(1.0, t_text + dt_text)

            if histories is not None:
                histories.append(torch.tensor([x_list], device=device, dtype=torch.long))

            # stop if both text and all images are done
            if t_text >= 1.0 - time_epsilon and all(
                float(img["t"]) >= 1.0 - time_epsilon for img in images
            ):
                break

        seq = torch.tensor([x_list], device=device, dtype=torch.long)
        if not return_dict:
            return seq

        return OneFlowSamplerOutput(
            sequences=seq,
            histories=histories,
            images=[img["latent"] for img in images] if images else None,
            image_times=[float(img["t"]) for img in images] if images else None,
        )

    @torch.no_grad()
    def infill(
        self,
        inputs: List[torch.Tensor | list],
        config: SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError("OneFlowSampler.infill is not implemented in v1.")


