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

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        with open(os.path.join(output_dir, "oneflow_config.json"), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.config), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "OneFlowModel":
        cfg_path = os.path.join(model_dir, "oneflow_config.json")
        sd_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config file: {cfg_path}")
        if not os.path.exists(sd_path):
            raise FileNotFoundError(f"Missing state dict file: {sd_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = OneFlowConfig(**json.load(f))
        model = cls(cfg)
        sd = torch.load(sd_path, map_location=map_location)
        model.load_state_dict(sd, strict=True)
        return model


