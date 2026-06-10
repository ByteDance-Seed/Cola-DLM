# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cola Text DiT — HuggingFace Transformers compatible, NA (no-padding) inference.

This module implements the **block-causal Diffusion Transformer prior**
``p_psi(z_0)`` of Cola DLM (Eq. 2.1.4 of the paper *Continuous Latent
Diffusion Language Model*, arXiv:2605.06548). It parameterizes the
vector field ``v_psi(z_t, t; z_0^(<b))`` of the continuous-flow prior
``z_0 = Phi^psi_{0<-1}(z_1)`` (Eq. 2.1.2), with the per-block
factorization

    p_psi(z_0) = p_psi(z_0^(1)) * prod_{b>=2} p_psi(z_0^(b) | z_0^(<b))

(Eq. 2.1.4) directly enforced by a block-causal attention pattern that
implements the visible set ``V_b = {sg(z_0^(<b)), z_t^(b)}`` of
Eq. 2.2.3.

In the Stage-2 pretraining pipeline, this prior is fit with the
conditional Flow Matching loss (Eq. 2.1.7 / 2.2.3)::

    L_FM = sum_{b=1..B} E_{t, z_0, z_1} [
        || v_psi(z_t^(b), t; z_0^(<b)) - u_t^(b)(z_0, z_1) ||_2^2
    ].

At inference time, each forward call realizes one Euler-style step of
the integrator ``Phi^psi_{0<-1}`` over the current generation block
(Eq. 2.2.5); composing many such steps along ``timesteps`` and many
blocks along the sequence axis produces the response latent
``hat z_0^(1:B)`` block-by-block. See ``docs/architecture.md`` for the
end-to-end mapping between paper notation and code paths.

After the NA refactor this model no longer accepts a padded
``(B, L_max, ...)`` batch. Instead, every entry point takes a
flatten-concat layout:

* ``txt: (L_q_total, c)`` — concatenated per-sample latents along a
  single sequence axis. During generation each per-sample slice is the
  current noisy block ``z_t^(b)``.
* ``txt_shape: (B, 1)`` — per-sample **K-side** length (prefix
  ``z^pre`` + previously committed clean blocks ``hat z_0^(<b)`` +
  current ``z_t^(b)``, i.e. the support of the visible set ``V_b``).
* ``txt_q_shape: (B, 1)`` — per-sample **Q-side** length. Either equal
  to ``txt_shape[i]`` (self-attention on the prefix) or ``block_size``
  (the block-wise generation main loop).

This removes the need for ``latent_labels`` inside attention (attention
masks are fully driven by ``txt_shape`` / ``txt_q_shape``) and for the
``shift_rope`` logic that handled label-3 pads.

The NA form strips all pad slots before flattening, so every sample's
K/Q are RoPE-indexed with canonical ``0``-start positions
(``k_offset = 0``, ``q_offset = txt_shape - txt_q_shape``). NA
inference therefore sees the same flattened layout as the original
trainer and does not need a separate pad-offset correction term.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

# Attention backend selection (optional, training speedup)
_ATTN_BACKEND = "naive"  # "naive", "sdpa", "flex"
_fused_flex_attn_func = None


def set_attn_backend(backend: str = "naive"):
    """Set attention backend: 'naive' (default), 'sdpa', or 'flex'."""
    global _ATTN_BACKEND, _fused_flex_attn_func
    assert backend in ("naive", "sdpa", "flex"), f"Unknown backend: {backend}"
    if backend == "flex":
        from torch.nn.attention.flex_attention import flex_attention
        @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
        def _fused(q, k, v, block_mask):
            return flex_attention(q, k, v, block_mask=block_mask)
        _fused_flex_attn_func = _fused
    _ATTN_BACKEND = backend

from .attention_utils import (
    create_na_block_causal_mask,
)
from .configuration_cola_dit import ColaDiTConfig

# ---------------------------------------------------------------------------
# Variable-length sequence helpers (NA / flatten-concat layout)
# ---------------------------------------------------------------------------


def _flatten(hid_list):
    """``List[Tensor(*_, c)]`` → ``(Tensor(L_total, c), txt_shape (B, 1))``."""
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid_list[0].device) for x in hid_list])
    hid = torch.cat([x.flatten(0, -2) for x in hid_list])
    return hid, shape


def _unflatten(hid, hid_shape):
    """Inverse of :func:`_flatten`: return a Python list of per-sample tensors."""
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    return [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]


# ---------------------------------------------------------------------------
# Timestep Embedding (pure PyTorch, no diffusers dependency)
# ---------------------------------------------------------------------------


def _get_sinusoidal_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Sinusoidal timestep embedding.

    Matches diffusers ``get_timestep_embedding`` with
    ``flip_sin_to_cos=False, downscale_freq_shift=0`` — i.e. the exponent
    denominator is ``half_dim``, NOT ``half_dim - 1``. This is the
    convention used by the original Cola trainer; using ``half_dim - 1``
    here would shift every frequency slightly and silently desync the
    entire AdaLN conditioning path.
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim  # downscale_freq_shift=0
    emb = torch.exp(exponent)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def initialize_weights(self):
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_hid.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

    def forward(self, timestep, device, dtype):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        emb = _get_sinusoidal_embedding(timestep, self.sinusoidal_dim).to(dtype)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        emb = self.proj_out(emb)
        return emb


# ---------------------------------------------------------------------------
# 1D Patch Embedding / Un-embedding (NA-aware via _unflatten/_flatten)
# ---------------------------------------------------------------------------


class PatchIn1D(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, dim)

    def initialize_weights(self):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, txt: torch.Tensor, txt_shape: torch.LongTensor):
        txt_shape_before_patchify = txt_shape
        if self.patch_size != 1:
            batch_list = _unflatten(txt, txt_shape)
            for i in range(len(batch_list)):
                batch_list[i] = rearrange(batch_list[i], "(T t) c -> T (t c)", t=self.patch_size)
            txt, txt_shape = _flatten(batch_list)
        txt = self.proj(txt)
        return txt, txt_shape, txt_shape_before_patchify


class PatchOut1D(nn.Module):
    def __init__(self, out_channels: int, patch_size: int, dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, out_channels * patch_size)

    def initialize_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, txt: torch.Tensor, txt_shape: torch.LongTensor, txt_shape_before_patchify):
        txt = self.proj(txt)
        if self.patch_size != 1:
            batch_list = _unflatten(txt, txt_shape)
            for i in range(len(batch_list)):
                batch_list[i] = rearrange(batch_list[i], "T (t c) -> (T t) c", t=self.patch_size)
            txt, txt_shape = _flatten(batch_list)
        return txt, txt_shape


# ---------------------------------------------------------------------------
# Text Rotary Embedding (lang-mode), freq computation is per-sample and cached
# ---------------------------------------------------------------------------

try:
    from rotary_embedding_torch import RotaryEmbedding as _RotaryEmbedding
    from rotary_embedding_torch import apply_rotary_emb
except ImportError as err:
    raise ImportError(
        "rotary-embedding-torch>=0.8.6 is required. " "Install with: pip install rotary-embedding-torch"
    ) from err


@lru_cache(maxsize=128)
def _get_axial_freqs(rope, dims, offsets=(), repeats=(), flatten=False):
    if len(offsets) == 0:
        offsets = (0,) * len(dims)
    Colon = slice(None)
    all_freqs = []
    for ind, dim in enumerate(dims):
        if rope.freqs_for == "lang":
            start = offsets[ind]
            end = start + int(dim)
            pos = torch.arange(start, end, device=rope.device)
        else:
            pos = torch.linspace(-1, 1, steps=dim, device=rope.device)
        freqs = rope.forward(pos, seq_len=dim, offset=offsets[ind])
        all_axis = [None] * len(dims)
        all_axis[ind] = Colon
        new_axis_slice = (Ellipsis, *all_axis, Colon)
        all_freqs.append(freqs[new_axis_slice])
    all_freqs = torch.broadcast_tensors(*all_freqs)
    all_freqs = torch.cat(all_freqs, dim=-1)
    if repeats:
        all_freqs = all_freqs.repeat(*repeats)
    if flatten:
        all_freqs = all_freqs.view(-1, all_freqs.size(-1))
    return all_freqs


class TextRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, rope_dim: int = 1):
        super().__init__()
        with torch.device("cpu"):
            self.rope = _RotaryEmbedding(dim=dim // rope_dim, freqs_for="lang", theta=10000)
            freqs = self.rope.freqs
            del self.rope.freqs
            self.rope.register_buffer("freqs", freqs.data)

    def apply_freqs(self, txt_q, txt_k, freqs_q, freqs_k):
        """Apply pre-computed ``freqs_q / freqs_k`` to ``txt_q / txt_k`` of
        shape ``(L, h, d)``.

        Not named ``apply`` because that would shadow :meth:`nn.Module.apply`,
        which breaks ``PreTrainedModel.post_init()`` (it recursively calls
        ``self.apply(self._initialize_weights)``).
        """
        txt_q = rearrange(txt_q, "L h d -> h L d")
        txt_k = rearrange(txt_k, "L h d -> h L d")
        txt_q = apply_rotary_emb(freqs_q, txt_q.float()).to(txt_q.dtype)
        txt_k = apply_rotary_emb(freqs_k, txt_k.float()).to(txt_k.dtype)
        txt_q = rearrange(txt_q, "h L d -> L h d")
        txt_k = rearrange(txt_k, "h L d -> L h d")
        return txt_q, txt_k

    def get_freqs_from_positions(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Build RoPE frequencies from explicit integer position IDs.

        ``position_ids`` is a 1-D ``(L,)`` long tensor of per-token
        positions — e.g. ``[0, 1, ..., L-1, 0, 1, ..., L-1]`` for the
        2L training trick where clean and noisy copies share positions.
        """
        pos = position_ids.float().to(self.rope.device)
        freqs = self.rope.forward(pos, seq_len=pos.shape[0], offset=0)
        return freqs

    def get_freqs(self, txt_shape, offset=None):
        """Concat per-sample RoPE frequencies along the flattened sequence.

        ``txt_shape`` is ``(B, 1)`` and ``offset`` is a Python list of
        per-sample starting positions (default ``[0] * B``).
        """
        if offset is None:
            offset = [0] * txt_shape.shape[0]
        txt_freq_list = []
        for i, l in enumerate(txt_shape[:, 0].tolist()):
            txt_freq = _get_axial_freqs(
                self.rope,
                (l,),
                offsets=(offset[i],),
                repeats=None,
                flatten=True,
            )
            txt_freq_list.append(txt_freq)
        return torch.cat(txt_freq_list, dim=0)


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN)
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    def __init__(self, dim: int, emb_dim: int, layers: list[str], modes: list[str] = None):
        super().__init__()
        if modes is None:
            modes = ["in", "out"]
        self.dim = dim
        self.layers = layers
        self.modes = modes
        for layer in layers:
            if "in" in modes:
                self.register_module(f"{layer}_in", nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)))
            if "out" in modes:
                self.register_module(f"{layer}_out", nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=True)))

    def initialize_weights(self):
        for layer in self.layers:
            for m in self.modes:
                nn.init.constant_(getattr(self, f"{layer}_{m}")[-1].weight, 0)
                nn.init.constant_(getattr(self, f"{layer}_{m}")[-1].bias, 0)

    def forward(self, hid, emb, layer, mode, hid_shape=None, norm_layer=None, residual=None, **kwargs):
        assert (mode in self.modes) and (layer in self.layers)
        # ``emb`` is either ``(L_total, d)`` (already flattened to match
        # ``hid``) or ``(B, d)`` (one per sample, needs to be repeated
        # according to ``hid_shape``).
        if hid.ndim > emb.ndim:
            emb = emb.unsqueeze(1) if emb.ndim == 1 else emb
        emb = getattr(self, f"{layer}_{mode}")(emb)

        if hid_shape is not None and emb.shape[0] != hid.shape[0]:
            hid_len = hid_shape.prod(-1)
            emb = torch.cat([e.repeat(l, *([1] * e.ndim)) for e, l in zip(emb, hid_len)])
        if mode == "in":
            shift, scale = emb.chunk(2, dim=-1)
            return norm_layer(hid) * (1 + scale) + shift
        if mode == "out":
            return hid * emb + residual
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MLP (GELU, tanh approximation)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, expand_ratio: int, **kwargs):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

    def forward(self, x):
        return self.proj_out(self.act(self.proj_in(x)))


# ---------------------------------------------------------------------------
# Text Attention (NA: flatten + cu_seqlens)
# ---------------------------------------------------------------------------


class ColaDiTAttention(nn.Module):
    def __init__(self, txt_dim, heads, head_dim, qk_bias, qk_norm_eps, rope_dim):
        super().__init__()
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = nn.Linear(txt_dim, qkv_dim, bias=qk_bias)
        self.proj_out = nn.Linear(inner_dim, txt_dim, bias=qk_bias)
        self.norm_q = nn.LayerNorm(head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.norm_k = nn.LayerNorm(head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.rope = TextRotaryEmbedding(dim=rope_dim)

        # Per-sample KV cache. Each entry is a ``(l_i_cum, h, d)`` tensor,
        # one per batch sample. ``None`` means "cache not populated yet".
        self._k_cache: Optional[list[torch.Tensor]] = None
        self._v_cache: Optional[list[torch.Tensor]] = None

    def set_kv_cache(self, flag: bool) -> None:
        self._k_cache = None
        self._v_cache = None

    def slow_attn(self, query, key, value, attn_mask=None, dropout_p=0.0):
        # Wrap the matmul in a bf16 autocast region so CUDA promotes
        # ``softmax`` to fp32 internally. Without this wrapper the bf16
        # softmax drifts from training-time numerics, and the error
        # compounds across diffusion steps and generation blocks.
        d_head = query.shape[-1]
        device_type = "cuda" if query.is_cuda else query.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            scale = 1.0 / (d_head**0.5)
            attn = query.mul(scale) @ key.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask.to(attn.dtype)
            attn_weight = attn.softmax(dim=-1)
            if dropout_p > 0:
                attn_weight = F.dropout(attn_weight, p=dropout_p, inplace=True)
            attn_out = attn_weight @ value
        return attn_out

    def forward(
        self,
        txt: torch.Tensor,
        *,
        txt_shape: torch.LongTensor,  # (B, 1) K length per sample
        txt_q_shape: torch.LongTensor,  # (B, 1) Q length per sample
        update_kv: bool = False,
        use_kv_cache: bool = False,
        attn_block_mask: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
        k_position_ids: Optional[torch.Tensor] = None,
        q_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ``txt`` is ``(L_q_total, d)`` — per-sample Q blocks already concatenated.
        txt_qkv = self.proj_qkv(txt)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        txt_q = self.norm_q(txt_q)
        txt_k = self.norm_k(txt_k)

        q_lens = txt_q_shape.flatten().tolist()

        # -- KV cache bookkeeping -------------------------------------------------
        new_ks = list(txt_k.split(q_lens))
        new_vs = list(txt_v.split(q_lens))

        if update_kv:
            if self._k_cache is None:
                self._k_cache = [k.clone() for k in new_ks]
                self._v_cache = [v.clone() for v in new_vs]
            else:
                self._k_cache = [torch.cat([c, n], dim=0) for c, n in zip(self._k_cache, new_ks)]
                self._v_cache = [torch.cat([c, n], dim=0) for c, n in zip(self._v_cache, new_vs)]
            full_k = torch.cat(self._k_cache, dim=0)
            full_v = torch.cat(self._v_cache, dim=0)
        elif use_kv_cache and self._k_cache is not None:
            full_k = torch.cat([torch.cat([c, n], dim=0) for c, n in zip(self._k_cache, new_ks)], dim=0)
            full_v = torch.cat([torch.cat([c, n], dim=0) for c, n in zip(self._v_cache, new_vs)], dim=0)
        else:
            full_k = txt_k
            full_v = txt_v

        # -- RoPE -----------------------------------------------------------------
        if k_position_ids is not None and q_position_ids is not None:
            freqs_k = self.rope.get_freqs_from_positions(k_position_ids)
            freqs_q = self.rope.get_freqs_from_positions(q_position_ids)
        else:
            k_offset = [0] * txt_shape.shape[0]
            q_offset = (txt_shape - txt_q_shape).flatten().int().tolist()
            freqs_k = self.rope.get_freqs(txt_shape, offset=k_offset)
            freqs_q = self.rope.get_freqs(txt_q_shape, offset=q_offset)
        txt_q, full_k = self.rope.apply_freqs(txt_q, full_k, freqs_q=freqs_q, freqs_k=freqs_k)

        # -- Attention ------------------------------------------------------------
        compute_dtype = torch.bfloat16 if txt_q.is_cuda else txt_q.dtype

        q_na = rearrange(txt_q.to(compute_dtype), "l h d -> 1 h l d")
        k_na = rearrange(full_k.to(compute_dtype), "l h d -> 1 h l d")
        v_na = rearrange(full_v.to(compute_dtype), "l h d -> 1 h l d")

        if _ATTN_BACKEND == "flex" and _fused_flex_attn_func is not None:
            from torch.nn.attention.flex_attention import BlockMask
            if isinstance(attn_block_mask, BlockMask):
                out = _fused_flex_attn_func(q_na, k_na, v_na, attn_block_mask)
            else:
                out = self.slow_attn(q_na, k_na, v_na, attn_mask=attn_block_mask)
        elif _ATTN_BACKEND == "sdpa":
            out = F.scaled_dot_product_attention(
                q_na, k_na, v_na,
                attn_mask=attn_block_mask,
                is_causal=False,
            )
        else:
            out = self.slow_attn(q_na, k_na, v_na, attn_mask=attn_block_mask)
        out = rearrange(out, "1 h l d -> l (h d)").type_as(txt_q)
        return self.proj_out(out)


# ---------------------------------------------------------------------------
# DiT Transformer Block
# ---------------------------------------------------------------------------


class ColaDiTBlock(nn.Module):
    def __init__(self, txt_dim, emb_dim, heads, head_dim, expand_ratio, norm_eps, qk_bias, rope_dim, block_size):
        super().__init__()
        self.msa_norm = nn.LayerNorm(txt_dim, eps=norm_eps, elementwise_affine=False)
        self.msa = ColaDiTAttention(
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm_eps=norm_eps,
            rope_dim=rope_dim,
        )
        self.mlp_norm = nn.LayerNorm(txt_dim, eps=norm_eps, elementwise_affine=False)
        self.mlp = MLP(dim=txt_dim, expand_ratio=expand_ratio)
        self.ada = AdaLN(dim=txt_dim, emb_dim=emb_dim, layers=["msa", "mlp"])

    def forward(
        self,
        txt,
        *,
        txt_shape,
        txt_q_shape,
        emb,
        update_kv=False,
        use_kv_cache=False,
        attn_block_mask=None,
        cpu_txt_shape=None,
        k_position_ids=None,
        q_position_ids=None,
    ):
        ada_kwargs = {"hid_shape": cpu_txt_shape if cpu_txt_shape is not None else txt_q_shape}

        txt_msa = self.ada(txt, emb=emb, layer="msa", mode="in", norm_layer=self.msa_norm, **ada_kwargs)
        txt_msa = self.msa(
            txt_msa,
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            update_kv=update_kv,
            use_kv_cache=use_kv_cache,
            attn_block_mask=attn_block_mask,
            k_position_ids=k_position_ids,
            q_position_ids=q_position_ids,
        )
        txt = self.ada(txt_msa, emb=emb, layer="msa", mode="out", residual=txt, **ada_kwargs)

        txt_mlp = self.ada(txt, emb=emb, layer="mlp", mode="in", norm_layer=self.mlp_norm, **ada_kwargs)
        txt_mlp = self.mlp(txt_mlp)
        txt = self.ada(txt_mlp, emb=emb, layer="mlp", mode="out", residual=txt, **ada_kwargs)
        return txt

    def set_kv_cache(self, flag):
        self.msa.set_kv_cache(flag)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColaDiTOutput:
    txt_sample: torch.Tensor


# ---------------------------------------------------------------------------
# Main Model: ColaDiTModel (PreTrainedModel)
# ---------------------------------------------------------------------------


class ColaDiTModel(PreTrainedModel):
    config_class = ColaDiTConfig
    base_model_prefix = "dit"
    supports_gradient_checkpointing = True

    def __init__(self, config: ColaDiTConfig):
        super().__init__(config)
        self.block_size = config.block_size
        self.heads = config.heads

        self.txt_in = PatchIn1D(
            in_channels=config.txt_in_channels,
            patch_size=config.patch_size,
            dim=config.txt_dim,
        )
        self.emb_in = TimestepEmbedding(
            sinusoidal_dim=256,
            hidden_dim=config.txt_dim,
            output_dim=config.emb_dim,
        )

        self.blocks = nn.ModuleList(
            [
                ColaDiTBlock(
                    txt_dim=config.txt_dim,
                    emb_dim=config.emb_dim,
                    heads=config.heads,
                    head_dim=config.head_dim,
                    expand_ratio=config.expand_ratio,
                    norm_eps=config.norm_eps,
                    qk_bias=config.qk_bias,
                    rope_dim=config.rope_dim,
                    block_size=config.block_size,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.txt_out_norm = nn.LayerNorm(config.txt_dim, eps=config.norm_eps, elementwise_affine=True)
        self.txt_out_ada = AdaLN(
            dim=config.txt_dim,
            emb_dim=config.emb_dim,
            layers=["out"],
            modes=["in"],
        )
        self.txt_out = PatchOut1D(
            out_channels=config.txt_out_channels,
            patch_size=config.patch_size,
            dim=config.txt_dim,
        )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        txt: torch.FloatTensor,
        txt_shape: torch.LongTensor,
        txt_q_shape: torch.LongTensor,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        update_kv: bool = False,
        use_kv_cache: bool = False,
        k_position_ids: Optional[torch.Tensor] = None,
        q_position_ids: Optional[torch.Tensor] = None,
        attn_mask_override: Optional[torch.Tensor] = None,
    ) -> ColaDiTOutput:
        """NA-form forward pass — one prior-transport step on block ``b``.

        Each call evaluates the block-causal vector field
        ``v_psi(z_t^(b), t; z_0^(<b))`` of Eq. 2.1.7 / 2.2.5 of the
        paper for every sample in the batch under the visible set
        ``V_b = {sg(z_0^(<b)), z_t^(b)}`` (Eq. 2.2.3). The integrator
        update ``z_{t-Δ}^(b) = z_t^(b) - Δ/T * v_psi(...)`` is performed
        by the caller (see :func:`generate_task_repaint_inference`).

        * ``txt``             : ``(L_q_total, c)`` — per-sample current
                                noisy block ``z_t^(b)``.
        * ``txt_shape``       : ``(B, 1)`` — per-sample K length. The
                                cumulative total of prefix ``z^pre`` +
                                previously committed clean blocks
                                ``hat z_0^(<b)`` + the current ``z_t^(b)``,
                                i.e. the support of ``V_b``.
        * ``txt_q_shape``     : ``(B, 1)`` — per-sample Q length.
                                ``block_size`` during block-wise generation;
                                ``txt_shape[i]`` during prefix self-attention.
        * ``timestep``        : Flow Matching time ``t``. Scalar,
                                ``(B,)``, or ``(L_q_total,)``.
        * ``update_kv``       : append ``txt_k, txt_v`` to the per-sample
                                prior KV cache (commit a freshly-generated
                                clean block ``hat z_0^(b)`` to history).
        * ``use_kv_cache``    : read the cached K/V. When ``update_kv``
                                is also True the append happens first,
                                then the read.

        * ``k_position_ids`` : ``(L_k_total,)`` optional explicit RoPE
                                positions for K. Used by the 2L training
                                trick to give clean and noisy copies
                                shared positions.
        * ``q_position_ids`` : ``(L_q_total,)`` optional explicit RoPE
                                positions for Q. Must be provided together
                                with ``k_position_ids``.

        When ``k_position_ids`` and ``q_position_ids`` are both ``None``
        (the default), RoPE uses ``k_offset = 0`` and
        ``q_offset = txt_shape - txt_q_shape`` — i.e. the same canonical
        NA layout as the original trainer and inference code.
        """
        txt, txt_shape_patched, txt_shape_before_patchify = self.txt_in(txt, txt_shape)

        # Expand ``timestep`` to per-token so AdaLN gets one embedding per
        # flattened position. Accept scalar / per-sample / per-token inputs.
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=txt.device, dtype=txt.dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and txt.shape[0] > 1:
            timestep = timestep.expand(txt.shape[0])
        elif timestep.shape[0] != txt.shape[0]:
            timestep = timestep.to(device=txt.device, dtype=txt.dtype)
        emb = self.emb_in(timestep, device=txt.device, dtype=txt.dtype)

        # Build NA attention mask once per forward — identical for every block.
        if attn_mask_override is not None:
            assert self.txt_in.patch_size == 1, (
                "attn_mask_override assumes patch_size=1; with patch_size>1 the "
                "mask dimensions would not match the patchified sequence lengths"
            )
            attn_mask = attn_mask_override
        else:
            attn_mask = create_na_block_causal_mask(
                txt_shape=txt_shape_patched,
                txt_q_shape=txt_q_shape,
                block_size=self.block_size,
                dtype=torch.bfloat16 if txt.is_cuda else txt.dtype,
                device=txt.device,
            )

        cpu_txt_shape = txt_q_shape.cpu()

        for block in self.blocks:
            txt = block(
                txt,
                txt_shape=txt_shape_patched,
                txt_q_shape=txt_q_shape,
                emb=emb,
                update_kv=update_kv,
                use_kv_cache=use_kv_cache,
                attn_block_mask=attn_mask,
                cpu_txt_shape=cpu_txt_shape,
                k_position_ids=k_position_ids,
                q_position_ids=q_position_ids,
            )

        if self.txt_out_norm is not None:
            txt = self.txt_out_ada(
                txt,
                emb=emb,
                layer="out",
                mode="in",
                hid_shape=cpu_txt_shape,
                norm_layer=self.txt_out_norm,
            )

        txt, _ = self.txt_out(txt, txt_shape_patched, txt_shape_before_patchify)
        return ColaDiTOutput(txt_sample=txt)


# Register with AutoConfig / AutoModel
AutoConfig.register("cola_dit", ColaDiTConfig)
AutoModel.register(ColaDiTConfig, ColaDiTModel)
