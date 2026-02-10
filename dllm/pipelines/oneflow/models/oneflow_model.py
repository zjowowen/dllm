from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OneFlowConfig:
    """
    Minimal config container for OneFlow.

    Note: We intentionally keep this independent from HuggingFace's PretrainedConfig for v1,
    because the OneFlow model will be a custom nn.Module using a vendored Transfusion backbone.
    """

    dim: int = 768
    vocab_size: int = 32000

    # token ids (to satisfy HF Trainer utilities like _align_special_tokens)
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None

    # multimodal (v1 supports a single modality type: image latents)
    dim_latent: int = 4  # image latent channel dim (e.g., SD VAE latents are 4)

    # Transfusion-style transformer trunk config
    depth: int = 12
    dim_head: int = 64
    heads: int = 12
    dropout: float = 0.0
    ff_expansion_factor: float = 4.0
    use_flex_attn: bool = False
    num_residual_streams: int = 1
    num_residual_fracs: int = 4

    # If True, tie to_q_logits.weight to text_embed.weight (weight tying).
    # Inspired by EditFlow's init_editflow_from_src which copies lm_head weights
    # to sub_logits/ins_logits heads for reduced cold-start instability.
    tie_q_logits_to_embedding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        # Mirror HF PretrainedConfig API used by Trainer integrations (e.g., W&B/TensorBoard).
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class OneFlowModel(nn.Module):
    """
    OneFlow unified multimodal model.

    This module will:
    - Embed text tokens + project image latent tokens into a unified sequence
    - Run a Transfusion-style transformer trunk with per-token time conditioning
    - Output OneFlow text insertion heads (pi, lambda_nonzero, Q) and image velocity head v
    """

    def __init__(self, config: OneFlowConfig):
        super().__init__()
        self.config = config

        try:
            from dllm.third_party.transfusion_pytorch.transfusion import (
                Transformer as TransfusionTransformer,
                derive_rotary_positions_from_modality_positions,
            )
            from rotary_embedding_torch import RotaryEmbedding
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing OneFlow optional dependencies for the Transfusion backbone.\n"
                "Please install the oneflow extra (to be added in this repo):\n"
                '  pip install -e ".[oneflow]"\n'
                "Or install the upstream deps from `reference/transfusion-pytorch/pyproject.toml`.\n"
                f"Original import error: {e}"
            ) from e

        self._derive_rotary_positions = derive_rotary_positions_from_modality_positions

        # embeddings
        self.text_embed = nn.Embedding(config.vocab_size, config.dim)
        self.latent_to_model = (
            nn.Identity()
            if config.dim_latent == config.dim
            else nn.Linear(config.dim_latent, config.dim)
        )

        # unified trunk (Transfusion-style)
        self.trunk = TransfusionTransformer(
            dim=config.dim,
            depth=config.depth,
            dim_head=config.dim_head,
            heads=config.heads,
            dropout=config.dropout,
            ff_expansion_factor=config.ff_expansion_factor,
            use_flex_attn=config.use_flex_attn,
            num_residual_streams=config.num_residual_streams,
            num_residual_fracs=config.num_residual_fracs,
        )

        # rotary embedding (positions are computed outside trunk)
        self.rotary_emb = RotaryEmbedding(self.trunk.dim_head)

        # text heads (pi / lambda / Q)
        self.to_pi = nn.Linear(config.dim, 1)
        self.to_lambda = nn.Linear(config.dim, 1)
        self.to_q_logits = nn.Linear(config.dim, config.vocab_size)

        # Optional weight tying: share to_q_logits.weight with text_embed.weight.
        # This is analogous to EditFlow's init_editflow_from_src which copies
        # lm_head weights to sub_logits/ins_logits, reducing cold-start instability.
        if getattr(config, "tie_q_logits_to_embedding", False):
            self.to_q_logits.weight = self.text_embed.weight

        # image head (velocity / flow in latent space)
        self.to_v = nn.Linear(config.dim, config.dim_latent, bias=False)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for OneFlowModel.

        Required inputs (v1):
        - input_ids: LongTensor [B, N]  (text ids; modality positions can be dummy ids)
        - attention_mask: LongTensor [B, N] (1 for valid tokens, 0 for padding). Optional.
        - is_any_modality: BoolTensor [B, N] (True for image-latent token positions). Optional if no modalities.
        - modality_tokens: FloatTensor [B, N, dim_latent] with non-zero values only at modality positions. Optional.
        - modality_positions: LongTensor [B, M, 3] containing (modality_type, offset, length). Optional.
        - times: FloatTensor [B, N] per-token times. Optional.

        Returns:
        - pi: [B, N]
        - lambda_nonzero: [B, N]
        - q_logits: [B, N, V]
        - v: [B, N, dim_latent]
        - hidden_states: [B, N, dim]
        """

        input_ids: torch.Tensor = kwargs["input_ids"]
        attention_mask: Optional[torch.Tensor] = kwargs.get("attention_mask", None)
        is_any_modality: Optional[torch.Tensor] = kwargs.get("is_any_modality", None)
        modality_tokens: Optional[torch.Tensor] = kwargs.get("modality_tokens", None)
        modality_positions: Optional[torch.Tensor] = kwargs.get("modality_positions", None)
        times: Optional[torch.Tensor] = kwargs.get("times", None)
        return_kv_cache: bool = bool(kwargs.get("return_kv_cache", False))
        cache: Optional[torch.Tensor] = kwargs.get("cache", None)
        decode_length: Optional[int] = kwargs.get("decode_length", None)

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,N], got {tuple(input_ids.shape)}")

        B, N = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((B, N), dtype=torch.long, device=device)
        if attention_mask.shape != (B, N):
            raise ValueError(
                f"attention_mask must be [B,N]={B,N}, got {tuple(attention_mask.shape)}"
            )

        if modality_tokens is None:
            modality_tokens = torch.zeros((B, N, self.config.dim_latent), device=device)
        if modality_tokens.shape[:2] != (B, N):
            raise ValueError(
                f"modality_tokens must be [B,N,*], got {tuple(modality_tokens.shape)}"
            )

        if is_any_modality is None:
            # Default: no modalities in the sequence
            is_any_modality = torch.zeros((B, N), dtype=torch.bool, device=device)
        is_any_modality = is_any_modality.to(torch.bool)
        if is_any_modality.shape != (B, N):
            raise ValueError(
                f"is_any_modality must be [B,N]={B,N}, got {tuple(is_any_modality.shape)}"
            )

        # embeddings
        safe_ids = input_ids.clamp_min(0)
        text_emb = self.text_embed(safe_ids)  # [B,N,dim]

        # project modality tokens to model dim, then mix with text embeddings
        mod_emb = self.latent_to_model(modality_tokens)  # [B,N,dim]
        tokens = torch.where(is_any_modality.unsqueeze(-1), mod_emb, text_emb)

        # per-token time (optional)
        if times is not None:
            if times.shape != (B, N):
                raise ValueError(f"times must be [B,N]={B,N}, got {tuple(times.shape)}")

        # build an attention mask that only masks KEYS (to avoid all-False rows -> NaNs)
        key_mask = attention_mask.to(torch.bool)  # [B,N]
        attn_mask = key_mask.unsqueeze(1).expand(B, N, N)  # [B, i, j]

        # rotary positions: if modality_positions provided, use transfusion's scheme; else plain 0..N-1
        if modality_positions is not None:
            modality_positions = modality_positions.to(device)
            rotary_pos = self._derive_rotary_positions(N, modality_positions)  # [B,N]
        else:
            rotary_pos = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        rotary_emb = self.rotary_emb(rotary_pos).unsqueeze(1)  # [B,1,N,dim_head]

        trunk_out = self.trunk(
            tokens,
            times=times,
            attn_mask=attn_mask,
            is_any_modality=is_any_modality,
            rotary_emb=rotary_emb,
            cache=cache,
            decode_length=decode_length,
            return_kv_cache=return_kv_cache,
        )

        if return_kv_cache:
            hidden, kv_cache = trunk_out
        else:
            hidden, kv_cache = trunk_out, None

        # heads
        pi = torch.sigmoid(self.to_pi(hidden)).squeeze(-1)
        lambda_nonzero = F.softplus(self.to_lambda(hidden)).squeeze(-1)
        q_logits = self.to_q_logits(hidden)
        v = self.to_v(hidden)

        out: Dict[str, Any] = {
            "pi": pi,
            "lambda_nonzero": lambda_nonzero,
            "q_logits": q_logits,
            "v": v,
            "hidden_states": hidden,
        }
        if return_kv_cache:
            out["kv_cache"] = kv_cache
        return out

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token-dependent modules to `new_num_tokens`.

        This mirrors the small piece of HF `PreTrainedModel.resize_token_embeddings`
        we need for OneFlowModel, since this model is a plain `nn.Module`.
        """
        new_num_tokens = int(new_num_tokens)
        if new_num_tokens <= 0:
            raise ValueError(f"new_num_tokens must be > 0, got {new_num_tokens}")

        # ---- embedding -----------------------------------------------------------
        old_embed = self.text_embed
        old_n = int(old_embed.num_embeddings)
        if new_num_tokens != old_n:
            new_embed = nn.Embedding(new_num_tokens, old_embed.embedding_dim)
            new_embed = new_embed.to(device=old_embed.weight.device, dtype=old_embed.weight.dtype)
            # copy existing weights
            n_copy = min(old_n, new_num_tokens)
            with torch.no_grad():
                new_embed.weight[:n_copy].copy_(old_embed.weight[:n_copy])
                if new_num_tokens > old_n:
                    nn.init.normal_(new_embed.weight[old_n:], mean=0.0, std=0.02)
            self.text_embed = new_embed

        # ---- output head (Q logits) ---------------------------------------------
        old_q = self.to_q_logits
        old_out = int(old_q.out_features)
        if new_num_tokens != old_out:
            new_q = nn.Linear(int(old_q.in_features), new_num_tokens, bias=True)
            new_q = new_q.to(device=old_q.weight.device, dtype=old_q.weight.dtype)
            n_copy = min(old_out, new_num_tokens)
            with torch.no_grad():
                new_q.weight[:n_copy].copy_(old_q.weight[:n_copy])
                new_q.bias[:n_copy].copy_(old_q.bias[:n_copy])
                if new_num_tokens > old_out:
                    nn.init.normal_(new_q.weight[old_out:], mean=0.0, std=0.02)
                    nn.init.zeros_(new_q.bias[old_out:])
            self.to_q_logits = new_q

        # keep config in sync
        self.config.vocab_size = int(new_num_tokens)

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        # Save a CPU state_dict for portability:
        # - avoids NPU-specific storages in `pytorch_model.bin`
        # - loads cleanly with newer PyTorch defaults (`weights_only=True`)
        # - allows inference/eval on CPU without requiring torch_npu/Ascend env
        sd = self.state_dict()
        sd_cpu = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}
        torch.save(sd_cpu, os.path.join(output_dir, "pytorch_model.bin"))
        with open(os.path.join(output_dir, "oneflow_config.json"), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.config), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_oneflow_config_path(model_dir: str) -> str:
        """
        Resolve `oneflow_config.json` for a directory.

        - Prefer `<model_dir>/oneflow_config.json`
        - If `model_dir` is an intermediate HF Trainer checkpoint (checkpoint-xxxx),
          allow falling back to:
            - sibling `<parent>/checkpoint-final/oneflow_config.json`
            - parent `<parent>/oneflow_config.json`
        """
        model_dir = str(model_dir)
        direct = os.path.join(model_dir, "oneflow_config.json")
        if os.path.exists(direct):
            return direct

        parent = os.path.dirname(os.path.abspath(model_dir))
        sibling_final = os.path.join(parent, "checkpoint-final", "oneflow_config.json")
        if os.path.exists(sibling_final):
            return sibling_final

        parent_cfg = os.path.join(parent, "oneflow_config.json")
        if os.path.exists(parent_cfg):
            return parent_cfg

        return direct  # for error message

    @staticmethod
    def _resolve_state_dict_path(model_dir: str) -> tuple[str, str]:
        """
        Resolve weights file path.

        Supports:
          - `pytorch_model.bin` (our `save_pretrained`)
          - `model.safetensors` (HF Trainer default when save_safetensors=True)

        Returns:
          (path, kind) where kind is "bin" or "safetensors".
        """
        model_dir = str(model_dir)
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return bin_path, "bin"
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(st_path):
            return st_path, "safetensors"
        # keep the old expectation in the error message
        return bin_path, "bin"

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "OneFlowModel":
        cfg_path = cls._resolve_oneflow_config_path(model_dir)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config file: {cfg_path}")

        sd_path, sd_kind = cls._resolve_state_dict_path(model_dir)
        if not os.path.exists(sd_path):
            raise FileNotFoundError(
                f"Missing state dict file: {sd_path} "
                f"(expected pytorch_model.bin or model.safetensors in {model_dir})"
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = OneFlowConfig(**json.load(f))
        model = cls(cfg)
        if sd_kind == "safetensors":
            try:
                from safetensors.torch import load_file as safe_load_file
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Found model.safetensors but safetensors is not available. "
                    "Please `pip install safetensors`."
                ) from e
            sd = safe_load_file(sd_path, device=str(map_location) if map_location is not None else "cpu")
        else:
            sd = torch.load(sd_path, map_location=map_location)
        model.load_state_dict(sd, strict=True)
        return model


