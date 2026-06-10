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
Shared attention utilities for Cola DLM (NA / flatten-concat).

This module provides the attention bookkeeping shared by the Text VAE
encoder/decoder (``ColaTextVAEModel``) and the block-causal DiT prior
(``ColaDiTModel``). In particular, :func:`create_na_block_causal_mask`
implements the **visible set**

    V_b = { sg(z_0^(<b)),  z_t^(b) }

of Eq. 2.2.3 of the paper *Continuous Latent Diffusion Language Model*
(arXiv:2605.06548), which defines the prior factorization

    p_psi(z_0) = p_psi(z_0^(1)) * prod_{b>=2} p_psi(z_0^(b) | z_0^(<b))

(Eq. 2.1.4). Concretely, the mask enforces the three properties of
``V_b``:

* **bidirectional within a block** — both Q and K positions inside the
  same generation block can attend to one another;
* **causal across blocks within a sample** — Q in block ``b_q`` may
  only attend to K in blocks ``b_k <= b_q``;
* **blocked across samples** — no cross-sample attention leakage in
  the NA layout.

The migration used to rely on a padded ``(B, L_max, ...)`` layout where
``latent_labels == 3`` marked padded positions. After the NA refactor
the whole inference pipeline now speaks "flatten-concat": each sample
contributes its own non-padded length ``n_i`` and tensors are
concatenated along a single ``L_total = sum(n_i)`` axis with a
companion ``txt_shape: (B, 1)`` describing per-sample lengths.

Remaining helpers below:

* :func:`cu_seqlens`, :func:`get_seqlen`, :func:`max_seqlen` — the same
  variable-length-sequence bookkeeping used on the training side.
* :func:`create_na_block_causal_mask` — additive block-diagonal
  block-causal mask for NA inference. Used by both the DiT prior and
  the VAE decoder, and is the implementation of ``V_b``.

The legacy ``create_attn_mask_from_labels`` /
``create_attn_mask_from_labels_one_block`` /
``create_block_causal_mask_naive`` /
``create_text_block_causal_attn_mask`` entry points have been removed;
they assumed a padded batch and no longer have any callers in the
migrated inference code.
"""

import torch
import torch.nn.functional as F


def cu_seqlens(seq_len: torch.LongTensor, skip_empty: bool = True) -> torch.Tensor:
    """Cumulative sequence lengths with a leading zero, ``int32``."""
    if skip_empty and (seq_len == 0).any():
        seq_len = seq_len[seq_len > 0]
    return F.pad(seq_len.cumsum(0), (1, 0)).int()


def get_seqlen(seq_shape: torch.LongTensor) -> torch.Tensor:
    """Flatten a ``(B, n)``-style shape tensor to per-sample length."""
    return seq_shape.prod(-1)


def max_seqlen(seq_len: torch.LongTensor) -> int:
    return seq_len.max().item()


def _as_flat_long(x: torch.LongTensor) -> torch.LongTensor:
    """Accept ``(B, 1)`` or ``(B,)`` per-sample length tensors uniformly."""
    return x.flatten() if x.ndim > 1 else x


def create_na_block_causal_mask(
    txt_shape: torch.LongTensor,
    txt_q_shape: torch.LongTensor,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive block-diagonal block-causal attention mask for NA sequences.

    Invariants the inference code guarantees and that we rely on here:

    * ``sum(txt_shape) == L_k_total`` where the K / V tensors fed to
      attention are the concatenation of per-sample K caches
      ``[k_0, k_1, ..., k_{B-1}]`` with ``k_i.shape[0] == txt_shape[i]``.
    * ``sum(txt_q_shape) == L_q_total`` laid out analogously.
    * Every ``txt_shape[i]`` is a positive multiple of ``block_size`` and
      every ``txt_q_shape[i]`` is either ``txt_shape[i]`` (self-attention
      on the prefix) or ``block_size`` (one-block query during block-wise
      generation).
    * Within a sample the last ``txt_q_shape[i]`` positions of K align to
      the last ``txt_q_shape[i]`` positions that Q refers to, i.e. Q is a
      suffix of K (which mirrors the original padded implementation,
      where ``txt_q_shape`` samples the tail of ``txt_shape``).

    Returns a ``(1, 1, L_q_total, L_k_total)`` additive mask with ``0``
    at allowed positions and ``dtype.min`` elsewhere, safe to add onto
    ``QK^T`` before the softmax.
    """
    k_lens = _as_flat_long(txt_shape)
    q_lens = _as_flat_long(txt_q_shape)
    assert k_lens.shape == q_lens.shape, (
        f"txt_shape {txt_shape.shape} and txt_q_shape {txt_q_shape.shape} " "must describe the same batch"
    )
    B = int(k_lens.shape[0])
    L_k = int(k_lens.sum().item())
    L_q = int(q_lens.sum().item())

    k_cu = F.pad(k_lens.cumsum(0), (1, 0))  # (B+1,)
    q_cu = F.pad(q_lens.cumsum(0), (1, 0))

    k_sample = torch.zeros(L_k, dtype=torch.long, device=device)
    k_local = torch.zeros(L_k, dtype=torch.long, device=device)
    q_sample = torch.zeros(L_q, dtype=torch.long, device=device)
    q_local = torch.zeros(L_q, dtype=torch.long, device=device)

    for b in range(B):
        k_len_b = int(k_lens[b].item())
        q_len_b = int(q_lens[b].item())
        if k_len_b > 0:
            k_sample[k_cu[b] : k_cu[b + 1]] = b
            k_local[k_cu[b] : k_cu[b + 1]] = torch.arange(k_len_b, device=device)
        if q_len_b > 0:
            q_sample[q_cu[b] : q_cu[b + 1]] = b
            # Q refers to the LAST ``q_len_b`` positions of K within the
            # same sample — this matches the original padded code where
            # ``txt_q_shape`` always described a suffix of ``txt_shape``.
            q_local[q_cu[b] : q_cu[b + 1]] = torch.arange(k_len_b - q_len_b, k_len_b, device=device)

    q_block = q_local.unsqueeze(1) // block_size  # (L_q, 1)
    k_block = k_local.unsqueeze(0) // block_size  # (1, L_k)
    same_sample = q_sample.unsqueeze(1) == k_sample.unsqueeze(0)  # (L_q, L_k)
    block_causal = q_block >= k_block  # (L_q, L_k)
    allowed = same_sample & block_causal

    if dtype.is_floating_point:
        min_val = torch.finfo(dtype).min
    else:
        min_val = torch.iinfo(dtype).min
    mask = torch.full((L_q, L_k), min_val, dtype=dtype, device=device)
    mask.masked_fill_(allowed, 0.0 if dtype.is_floating_point else 0)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L_q, L_k)


def create_2l_block_causal_mask(
    txt_shape: torch.LongTensor,
    txt_q_shape: torch.LongTensor,
    seq_lens: list[int],
    block_size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Attention mask for the 2L training trick.

    Each sample has layout ``[clean(L) | noisy(L)]``.  Q may cover the
    full ``2L`` (when ``txt_q_shape == txt_shape``) or only the noisy
    tail ``L`` (when ``txt_q_shape[i] == seq_lens[i]``).  The mask
    enforces, for **noisy** Q positions:

    * Noisy Q at sequence block ``b`` can attend to clean K blocks
      ``0 .. b-1`` (strict causal — clean block ``b`` excluded).
    * Noisy Q at block ``b`` can attend to its own noisy K block ``b``
      (bidirectional within block).
    * No attention to other noisy K blocks.

    For **clean** Q positions (only present when ``txt_q_shape == 2L``):

    * Standard block-causal over the clean copy.
    * No attention to the noisy copy.

    Parameters
    ----------
    txt_shape : (B, 1)
        Per-sample K length, equal to ``2 * seq_lens[i]``.
    txt_q_shape : (B, 1)
        Per-sample Q length — either ``2 * seq_lens[i]`` or ``seq_lens[i]``.
    seq_lens : list[int]
        Per-sample original (single-copy) sequence length ``L``.
    block_size : int or list[int]
        Block size for block-causal attention. If list, per-sample block sizes.
    dtype, device

    Returns
    -------
    (1, 1, L_q_total, L_k_total) additive mask.
    """
    k_lens = _as_flat_long(txt_shape)
    q_lens = _as_flat_long(txt_q_shape)
    B = int(k_lens.shape[0])
    L_k = int(k_lens.sum().item())
    L_q = int(q_lens.sum().item())

    if isinstance(block_size, int):
        block_sizes = [block_size] * B
    else:
        block_sizes = list(block_size)

    k_cu = F.pad(k_lens.cumsum(0), (1, 0))
    q_cu = F.pad(q_lens.cumsum(0), (1, 0))

    k_sample = torch.zeros(L_k, dtype=torch.long, device=device)
    k_block_idx = torch.zeros(L_k, dtype=torch.long, device=device)
    k_is_clean = torch.zeros(L_k, dtype=torch.bool, device=device)
    q_sample = torch.zeros(L_q, dtype=torch.long, device=device)
    q_block_idx = torch.zeros(L_q, dtype=torch.long, device=device)
    q_is_clean = torch.zeros(L_q, dtype=torch.bool, device=device)

    for b in range(B):
        L_i = seq_lens[b]
        bs_i = block_sizes[b]
        q_len_b = int(q_lens[b].item())

        k_sample[k_cu[b] : k_cu[b + 1]] = b
        k_local = torch.arange(L_i, device=device)
        k_block_idx[k_cu[b] : k_cu[b] + L_i] = k_local // bs_i
        k_block_idx[k_cu[b] + L_i : k_cu[b + 1]] = k_local // bs_i
        k_is_clean[k_cu[b] : k_cu[b] + L_i] = True

        q_sample[q_cu[b] : q_cu[b + 1]] = b
        if q_len_b == 2 * L_i:
            q_local = torch.arange(L_i, device=device)
            q_block_idx[q_cu[b] : q_cu[b] + L_i] = q_local // bs_i
            q_block_idx[q_cu[b] + L_i : q_cu[b + 1]] = q_local // bs_i
            q_is_clean[q_cu[b] : q_cu[b] + L_i] = True
        else:
            q_local = torch.arange(q_len_b, device=device)
            q_block_idx[q_cu[b] : q_cu[b + 1]] = q_local // bs_i

    q_blk = q_block_idx.unsqueeze(1)  # (L_q, 1)
    k_blk = k_block_idx.unsqueeze(0)  # (1, L_k)
    same_sample = q_sample.unsqueeze(1) == k_sample.unsqueeze(0)

    k_clean = k_is_clean.unsqueeze(0)
    q_clean = q_is_clean.unsqueeze(1)

    noisy_q_clean_k = (~q_clean) & k_clean & (q_blk > k_blk)
    noisy_q_noisy_k = (~q_clean) & (~k_clean) & (q_blk == k_blk)
    clean_q_clean_k = q_clean & k_clean & (q_blk >= k_blk)

    allowed = same_sample & (noisy_q_clean_k | noisy_q_noisy_k | clean_q_clean_k)

    if dtype.is_floating_point:
        min_val = torch.finfo(dtype).min
    else:
        min_val = torch.iinfo(dtype).min
    mask = torch.full((L_q, L_k), min_val, dtype=dtype, device=device)
    mask.masked_fill_(allowed, 0.0 if dtype.is_floating_point else 0)
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# FlexAttention block mask (Fast-dLLM v2 style, adapted for NA form)
# ---------------------------------------------------------------------------

def create_2l_flex_block_mask(
    txt_shape: torch.LongTensor,
    txt_q_shape: torch.LongTensor,
    seq_lens: list[int],
    block_size,
    device: torch.device,
):
    """FlexAttention block mask for the 2L training trick (NA form).

    Uses Fast-dLLM v2 layout ``[xt(noisy, L) | x0(clean, L)]`` per sample.
    The caller must construct the extended sequence in this order.

    Ported directly from Fast-dLLM v2 ``block_diff_mask`` (modeling.py:44-87):

    * M_BD:  bidirectional within same block AND same copy type
    * M_OBC: noisy Q → clean K, strict causal (``q_block > k_block``)
    * M_BC:  clean Q → clean K, block-causal (``q_block >= k_block``)

    Adapted for Cola's NA form (multiple samples concatenated, batch=1)
    by adding per-position ``sample_id`` to prevent cross-sample attention.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    k_lens = _as_flat_long(txt_shape)
    q_lens = _as_flat_long(txt_q_shape)
    B = int(k_lens.shape[0])

    if isinstance(block_size, int):
        block_sizes = [block_size] * B
    else:
        block_sizes = list(block_size)

    L_k = int(k_lens.sum().item())
    L_q = int(q_lens.sum().item())

    k_cu = F.pad(k_lens.cumsum(0), (1, 0))
    q_cu = F.pad(q_lens.cumsum(0), (1, 0))

    # Per-position metadata (O(n) memory)
    # Layout per sample: [xt(noisy, L_i) | x0(clean, L_i)]
    # x0_flag = True for clean positions (second half, idx >= L_i within sample)
    q_sample_id = torch.zeros(L_q, dtype=torch.int32, device=device)
    q_x0_flag = torch.zeros(L_q, dtype=torch.bool, device=device)
    q_block_idx = torch.zeros(L_q, dtype=torch.int32, device=device)
    k_sample_id = torch.zeros(L_k, dtype=torch.int32, device=device)
    k_x0_flag = torch.zeros(L_k, dtype=torch.bool, device=device)
    k_block_idx = torch.zeros(L_k, dtype=torch.int32, device=device)

    for b_idx in range(B):
        L_i = seq_lens[b_idx]
        bs_i = block_sizes[b_idx]

        # K side: [noisy(L_i) | clean(L_i)]
        k_start = int(k_cu[b_idx].item())
        k_local = torch.arange(L_i, device=device)
        k_sample_id[k_start: k_start + 2 * L_i] = b_idx
        k_x0_flag[k_start + L_i: k_start + 2 * L_i] = True  # second half = clean
        k_block_idx[k_start: k_start + L_i] = k_local // bs_i
        k_block_idx[k_start + L_i: k_start + 2 * L_i] = k_local // bs_i

        # Q side
        q_start = int(q_cu[b_idx].item())
        q_len_b = int(q_lens[b_idx].item())
        q_sample_id[q_start: q_start + q_len_b] = b_idx
        if q_len_b == 2 * L_i:
            q_local = torch.arange(L_i, device=device)
            q_x0_flag[q_start + L_i: q_start + 2 * L_i] = True
            q_block_idx[q_start: q_start + L_i] = q_local // bs_i
            q_block_idx[q_start + L_i: q_start + 2 * L_i] = q_local // bs_i
        else:
            q_local = torch.arange(q_len_b, device=device)
            q_block_idx[q_start: q_start + q_len_b] = q_local // bs_i

    def mask_mod(b, h, q_idx, kv_idx):
        same = q_sample_id[q_idx] == k_sample_id[kv_idx]
        q_x0 = q_x0_flag[q_idx]
        k_x0 = k_x0_flag[kv_idx]
        q_b = q_block_idx[q_idx]
        k_b = k_block_idx[kv_idx]
        M_BD = (q_b == k_b) & (q_x0 == k_x0)
        M_OBC = (q_b > k_b) & k_x0 & (~q_x0)
        M_BC = (q_b >= k_b) & k_x0 & q_x0
        return same & (M_BD | M_OBC | M_BC)

    return create_block_mask(mask_mod, B=1, H=None, Q_LEN=L_q, KV_LEN=L_k, device=device)
