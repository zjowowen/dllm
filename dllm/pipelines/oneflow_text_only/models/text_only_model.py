from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bias_dropout_add_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _LayerNorm(nn.Module):
    """
    Reference DDiT-style LayerNorm without bias, keeping fp32 normalize path.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x.float(), [self.dim])
        return out * self.weight[None, None, :]


class _TimestepEmbedder(nn.Module):
    """
    Reference text example style timestep embedder.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(time: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=time.device
        )
        args = time[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        return self.mlp(t_freq)


class _Rotary(nn.Module):
    """
    From: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
    """

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def get_cos_sin(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)

            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached.to(dtype=dtype), self.sin_cached.to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    From: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L20
    """
    # cos and sin are generated with shape [1, seqlen, 3, 1, dim]
    # We want them for [B, H, S, D] but x here is transposed? No, x is [B, H, S, D] in DDiTBlock?
    # wait, the error trace:
    # q = self.qw(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,S,D]
    # In DDiTBlock we pass q, k directly to _apply_rotary_emb.
    # So x has shape [B, H, S, D]

    # Original FB code:
    # cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]  # results in shape [S, dim/2]
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]

    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    # The issue: repeat_interleave over dim=-1 makes it [S, D]
    # But x is [B, H, S, D]. So we need to match dimensions for broadcasting.
    # original FB code used einops: repeat(cos, "... d -> ... 1 (2 d)")
    # let's be explicitly clear with unsqueeze
    cos = cos.repeat_interleave(2, dim=-1)  # shape [S, D]
    sin = sin.repeat_interleave(2, dim=-1)  # shape [S, D]

    # Add dimensions for [B, H, S, D]
    cos = cos.unsqueeze(0).unsqueeze(1)  # shape [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(1)  # shape [1, 1, S, D]

    return x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin


class _DDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = int(n_heads)
        self.dim = int(dim)
        self.head_dim = self.dim // self.n_heads
        self.dropout = float(dropout)

        self.norm1 = _LayerNorm(dim=self.dim)
        self.qw = nn.Linear(self.dim, self.dim, bias=False)
        self.kw = nn.Linear(self.dim, self.dim, bias=False)
        self.vw = nn.Linear(self.dim, self.dim, bias=False)
        self.attn_out = nn.Linear(self.dim, self.dim, bias=False)

        self.norm2 = _LayerNorm(dim=self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_ratio * self.dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * self.dim, self.dim, bias=True),
        )

        self.ada_ln = nn.Linear(cond_dim, 6 * self.dim, bias=True)
        self.ada_ln.weight.data.zero_()
        self.ada_ln.bias.data.zero_()

    def forward(
        self,
        *,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen = x.shape[0], x.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond)[:, None].chunk(6, dim=2)

        x_skip = x
        x = _modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,S,D]
        k = self.kw(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.vw(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Keep rotary math in fp32 for stability.
        q_dtype = q.dtype
        k_dtype = k.dtype
        q = _apply_rotary_emb(q.float(), rotary_cos.float(), rotary_sin.float()).to(q_dtype)
        k = _apply_rotary_emb(k.float(), rotary_cos.float(), rotary_sin.float()).to(k_dtype)

        # Key-only masking to avoid all-False rows.
        attn_bias = torch.where(
            key_mask[:, None, None, :],
            torch.zeros((), device=x.device, dtype=q.dtype),
            torch.full((), -1e9, device=x.device, dtype=q.dtype),
        )
        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # [B,H,S,D]
        x_attn = x_attn.transpose(1, 2).reshape(bsz, seqlen, self.dim)

        x = _bias_dropout_add_scale(
            x=self.attn_out(x_attn),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )
        x = _bias_dropout_add_scale(
            x=self.mlp(_modulate(self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )
        return x


@dataclass
class OneFlowTextOnlyConfig:
    """
    Config for text-only OneFlow pipeline using a reference-style DDiT backbone.
    """

    vocab_size: int = 32000
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None

    # Backbone (reference text model style)
    hidden_size: int = 768
    cond_dim: int = 128
    n_blocks: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    mlp_ratio: int = 4
    rotary_base: int = 10_000

    # Keep compatibility with OneFlow trainer/sampler output contract.
    dim_latent: int = 4
    tie_q_logits_to_embedding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class OneFlowTextOnlyModel(nn.Module):
    """
    Text-only model with reference-style DDiT blocks and OneFlow-compatible heads.
    """

    def __init__(self, config: OneFlowTextOnlyConfig):
        super().__init__()
        self.config = config

        self.vocab_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_embedding = _TimestepEmbedder(hidden_size=config.cond_dim)
        self.rotary = _Rotary(dim=config.hidden_size // config.n_heads, base=int(config.rotary_base))

        self.blocks = nn.ModuleList(
            [
                _DDiTBlock(
                    dim=config.hidden_size,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        self.norm_final = _LayerNorm(config.hidden_size)
        self.final_ada_ln = nn.Linear(config.cond_dim, 2 * config.hidden_size, bias=True)
        self.final_ada_ln.weight.data.zero_()
        self.final_ada_ln.bias.data.zero_()

        self.to_pi = nn.Linear(config.hidden_size, 1)
        self.to_lambda = nn.Linear(config.hidden_size, 1)
        self.to_q_logits = nn.Linear(config.hidden_size, config.vocab_size)
        if bool(getattr(config, "tie_q_logits_to_embedding", False)):
            self.to_q_logits.weight = self.vocab_embed.weight
        self.to_v = nn.Linear(config.hidden_size, config.dim_latent, bias=False)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        input_ids: torch.Tensor = kwargs["input_ids"]
        attention_mask: Optional[torch.Tensor] = kwargs.get("attention_mask", None)
        times: Optional[torch.Tensor] = kwargs.get("times", None)
        return_kv_cache: bool = bool(kwargs.get("return_kv_cache", False))

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,N], got {tuple(input_ids.shape)}")
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seqlen), dtype=torch.long, device=device)
        if attention_mask.shape != (bsz, seqlen):
            raise ValueError(
                f"attention_mask must be [B,N]={bsz,seqlen}, got {tuple(attention_mask.shape)}"
            )

        if times is None:
            times = torch.zeros((bsz, seqlen), dtype=torch.float32, device=device)
        if times.shape != (bsz, seqlen):
            raise ValueError(f"times must be [B,N]={bsz,seqlen}, got {tuple(times.shape)}")

        ids = input_ids.clamp_min(0)
        x = self.vocab_embed(ids)

        # Reference design conditions on one scalar time per sample.
        # For per-token time inputs, we use sample-level mean as robust fallback.
        sample_time = times.float().mean(dim=1)  # [B]
        cond = F.silu(self.time_embedding(sample_time))

        key_mask = attention_mask.to(torch.bool)
        rotary_cos, rotary_sin = self.rotary.get_cos_sin(
            seq_len=seqlen,
            device=device,
            dtype=x.dtype,
        )

        enable_amp = device.type in {"cuda", "npu"}
        amp_dtype = torch.bfloat16 if enable_amp else torch.float32
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enable_amp):
            for blk in self.blocks:
                x = blk(
                    x=x,
                    cond=cond,
                    key_mask=key_mask,
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin,
                )
            shift, scale = self.final_ada_ln(cond)[:, None].chunk(2, dim=2)
            hidden = _modulate(self.norm_final(x), shift=shift, scale=scale)

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
            out["kv_cache"] = None
        return out

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        new_num_tokens = int(new_num_tokens)
        if new_num_tokens <= 0:
            raise ValueError(f"new_num_tokens must be > 0, got {new_num_tokens}")

        old_embed = self.vocab_embed
        old_n = int(old_embed.num_embeddings)
        if new_num_tokens != old_n:
            new_embed = nn.Embedding(new_num_tokens, old_embed.embedding_dim)
            new_embed = new_embed.to(device=old_embed.weight.device, dtype=old_embed.weight.dtype)
            n_copy = min(old_n, new_num_tokens)
            with torch.no_grad():
                new_embed.weight[:n_copy].copy_(old_embed.weight[:n_copy])
                if new_num_tokens > old_n:
                    nn.init.normal_(new_embed.weight[old_n:], mean=0.0, std=0.02)
            self.vocab_embed = new_embed

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

        self.config.vocab_size = int(new_num_tokens)

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.state_dict()
        state_dict_cpu = {
            k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in state_dict.items()
        }
        torch.save(state_dict_cpu, os.path.join(output_dir, "pytorch_model.bin"))

        cfg = dataclasses.asdict(self.config)
        with open(os.path.join(output_dir, "oneflow_text_only_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        # Keep compatibility with existing eval/trainer tooling.
        with open(os.path.join(output_dir, "oneflow_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_config_path(model_dir: str) -> str:
        model_dir = str(model_dir)
        candidates = [
            os.path.join(model_dir, "oneflow_text_only_config.json"),
            os.path.join(model_dir, "oneflow_config.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        parent = os.path.dirname(os.path.abspath(model_dir))
        parent_candidates = [
            os.path.join(parent, "checkpoint-final", "oneflow_text_only_config.json"),
            os.path.join(parent, "checkpoint-final", "oneflow_config.json"),
            os.path.join(parent, "oneflow_text_only_config.json"),
            os.path.join(parent, "oneflow_config.json"),
        ]
        for p in parent_candidates:
            if os.path.exists(p):
                return p

        return candidates[0]

    @staticmethod
    def _resolve_state_dict_path(model_dir: str) -> tuple[str, str]:
        model_dir = str(model_dir)
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return bin_path, "bin"
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(st_path):
            return st_path, "safetensors"
        return bin_path, "bin"

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "OneFlowTextOnlyModel":
        cfg_path = cls._resolve_config_path(model_dir)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config file: {cfg_path}")

        sd_path, sd_kind = cls._resolve_state_dict_path(model_dir)
        if not os.path.exists(sd_path):
            raise FileNotFoundError(
                f"Missing state dict file: {sd_path} "
                f"(expected pytorch_model.bin or model.safetensors in {model_dir})"
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = OneFlowTextOnlyConfig(**json.load(f))
        model = cls(cfg)

        if sd_kind == "safetensors":
            try:
                from safetensors.torch import load_file as safe_load_file
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "Found model.safetensors but safetensors is not available. "
                    "Please install `safetensors`."
                ) from exc
            sd = safe_load_file(sd_path, device=str(map_location) if map_location is not None else "cpu")
        else:
            sd = torch.load(sd_path, map_location=map_location)

        model.load_state_dict(sd, strict=True)
        return model

