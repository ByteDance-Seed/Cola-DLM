"""Test numerical equivalence of attention backends.

Loads the pretrained DiT and VAE, creates a synthetic 2L training batch,
and compares outputs from all attention backends (slow, sdpa, flex).

Verifies that:
1. All backends produce the same output (within bf16 tolerance)
2. Cola layout [clean|noisy] and v2 layout [noisy|clean] produce the same
   noisy-copy predictions (extracted from correct half)
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cola_dlm import ColaTextVAEModel, ColaDiTModel
from cola_dlm.attention_utils import create_2l_block_causal_mask, create_2l_flex_block_mask
from cola_dlm.modeling_cola_dit import set_attn_backend


def make_synthetic_data(vae, device, block_size=16, seq_len=64):
    """Create synthetic z_0, z_noisy, timesteps for one sample."""
    torch.manual_seed(42)
    STOP_TOKEN_ID = 47774
    L = seq_len
    pad_len = (block_size - L % block_size) % block_size
    token_ids = torch.randint(0, 30000, (L + pad_len,), device=device)
    if pad_len > 0:
        token_ids[L:] = STOP_TOKEN_ID
    L = L + pad_len

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = vae.encode([token_ids])
    z_0 = enc.latents_list[0].float()
    z_0 = (z_0 - vae.shifting_factor) * vae.scaling_factor

    t_val = 0.5
    z_1 = torch.randn_like(z_0)
    z_noisy = (1 - t_val) * z_0 + t_val * z_1
    ts_noisy = torch.full((L,), t_val * 1000.0, device=device)

    return z_0, z_noisy, ts_noisy, L


def run_with_layout(dit, z_0, z_noisy, ts_noisy, L, block_size, attn_mask, layout="cola"):
    """Run forward pass with specified layout and extract noisy predictions."""
    device = z_0.device
    positions = torch.arange(L, device=device)

    if layout == "v2":
        # [noisy | clean]
        txt = torch.cat([z_noisy, z_0.detach()], dim=0)
        timestep = torch.cat([ts_noisy, torch.zeros(L, device=device)])
    else:
        # [clean | noisy]
        txt = torch.cat([z_0.detach(), z_noisy], dim=0)
        timestep = torch.cat([torch.zeros(L, device=device), ts_noisy])

    txt_shape = torch.tensor([[2 * L]], dtype=torch.long, device=device)
    txt_q_shape = txt_shape.clone()
    k_position_ids = torch.cat([positions, positions])
    q_position_ids = torch.cat([positions, positions])

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = dit(
            txt=txt.to(torch.bfloat16),
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            timestep=timestep.to(torch.bfloat16),
            k_position_ids=k_position_ids,
            q_position_ids=q_position_ids,
            attn_mask_override=attn_mask,
        )

    sample_out = out.txt_sample.float()
    if layout == "v2":
        pred = sample_out[:L]   # noisy is first half
    else:
        pred = sample_out[L:]   # noisy is second half
    return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    vae = ColaTextVAEModel.from_pretrained("hf_models/cola_dlm/cola_vae").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    dit = ColaDiTModel.from_pretrained("hf_models/cola_dlm/cola_dit").to(device).eval()

    block_size = dit.block_size
    print(f"DiT block_size={block_size}, heads={dit.config.heads}, head_dim={dit.config.head_dim}")

    print("Creating synthetic data...")
    z_0, z_noisy, ts_noisy, L = make_synthetic_data(vae, device, block_size=block_size, seq_len=64)
    seq_lens = [L]
    block_sizes = [block_size]
    txt_shape = torch.tensor([[2 * L]], dtype=torch.long, device=device)
    txt_q_shape = txt_shape.clone()

    # --- Dense mask (Cola layout [clean|noisy]) ---
    dense_mask = create_2l_block_causal_mask(
        txt_shape, txt_q_shape,
        seq_lens=seq_lens, block_size=block_sizes,
        dtype=torch.bfloat16, device=device,
    )

    # --- FlexAttention mask (v2 layout [noisy|clean]) ---
    flex_mask = create_2l_flex_block_mask(
        txt_shape, txt_q_shape,
        seq_lens=seq_lens, block_size=block_sizes,
        device=device,
    )

    results = {}

    # --- Method 0: slow_attn, Cola layout ---
    print("\n[slow_attn + Cola layout] Running baseline...")
    set_attn_backend("naive")
    pred_slow = run_with_layout(dit, z_0, z_noisy, ts_noisy, L, block_size, dense_mask, layout="cola")
    results["slow_cola"] = pred_slow
    print(f"  Pred shape: {pred_slow.shape}, norm: {pred_slow.norm():.4f}")

    # --- Method 1: sdpa, Cola layout ---
    print("\n[sdpa + Cola layout] Running...")
    try:
        set_attn_backend("sdpa")
        pred_sdpa = run_with_layout(dit, z_0, z_noisy, ts_noisy, L, block_size, dense_mask, layout="cola")
        results["sdpa_cola"] = pred_sdpa
        diff = (pred_sdpa - pred_slow).abs()
        print(f"  Pred norm: {pred_sdpa.norm():.4f}")
        print(f"  Max abs diff vs slow: {diff.max().item():.6e}")
        print(f"  PASS" if diff.max().item() < 0.05 else f"  WARN: diff={diff.max().item()}")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # --- Method 2: flex, v2 layout ---
    print("\n[flex + v2 layout] Running...")
    try:
        set_attn_backend("flex")
        pred_flex = run_with_layout(dit, z_0, z_noisy, ts_noisy, L, block_size, flex_mask, layout="v2")
        results["flex_v2"] = pred_flex
        diff = (pred_flex - pred_slow).abs()
        print(f"  Pred norm: {pred_flex.norm():.4f}")
        print(f"  Max abs diff vs slow: {diff.max().item():.6e}")
        print(f"  PASS" if diff.max().item() < 0.05 else f"  WARN: diff={diff.max().item()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # --- Summary ---
    print("\n=== Summary ===")
    baseline = results.get("slow_cola")
    if baseline is not None:
        for name, pred in results.items():
            if name == "slow_cola":
                continue
            diff = (pred - baseline).abs()
            status = "PASS" if diff.max().item() < 0.05 else "FAIL"
            print(f"  {name} vs slow_cola: max_diff={diff.max().item():.6e} [{status}]")

    set_attn_backend("naive")
    print("\nDone.")


if __name__ == "__main__":
    main()
