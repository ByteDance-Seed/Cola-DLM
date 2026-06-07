"""
Train Cola-DLM DiT from scratch on ClimbMix-400B data.

Freezes a pretrained VAE, randomly initializes a DiT, and trains it using
flow matching with the 2L trick. Uses Muon optimizer for matrix params and
AdamW for scalars/embeddings (nanochat-style), with auto-computed batch size
and training horizon.

Single-GPU:
    python scripts/cola_pretrain.py --num-iterations=100 --run=dummy

Multi-GPU (DistMuonAdamW handles gradient sync — no DDP wrapper):
    torchrun --standalone --nproc_per_node=8 scripts/cola_pretrain.py --run=my_run
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cola-DLM Train DiT from Scratch")
# Model architecture
parser.add_argument("--dit-num-layers", type=int, default=24)
parser.add_argument("--dit-txt-dim", type=int, default=2048)
parser.add_argument("--dit-heads", type=int, default=16)
parser.add_argument("--dit-head-dim", type=int, default=128)
parser.add_argument("--dit-expand-ratio", type=int, default=4)
parser.add_argument("--dit-block-size", type=int, default=16)
# VAE (frozen, pretrained)
parser.add_argument("--vae-path", type=str, default="hf_models/cola_dlm/cola_vae")
parser.add_argument("--tokenizer-path", type=str, default="hf_models/tokenizer.json")
# Output
parser.add_argument("--output-dir", type=str, default="cola_pretrain_checkpoints")
parser.add_argument("--run", type=str, default="dummy")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="-1 = auto from target-param-data-ratio")
parser.add_argument("--target-param-data-ratio", type=float, default=12,
                    help="Tokens-to-params ratio (Chinchilla=20)")
# Batch
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--total-batch-size", type=int, default=-1, help="-1 = auto from scaling law")
parser.add_argument("--max-seq-len", type=int, default=512)
# Optimizer
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR for 2D params")
parser.add_argument("--scalar-lr", type=float, default=0.3, help="AdamW LR for 1D/scalar params")
parser.add_argument("--weight-decay", type=float, default=0.28, help="Muon weight decay")
parser.add_argument("--grad-clip", type=float, default=1.0)
# Schedule
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
# Flow matching
parser.add_argument("--timestep-dist", type=str, default="logit_normal", choices=["logit_normal", "uniform"])
parser.add_argument("--logit-normal-loc", type=float, default=0.0)
parser.add_argument("--logit-normal-scale", type=float, default=1.0)
parser.add_argument("--T", type=float, default=1000.0)
# VAE
parser.add_argument("--vae-mode", type=str, default="sample", choices=["sample", "mode"])
# Block size randomization
parser.add_argument("--block-size-probs", type=str, default=None)
# Simulated prompt-response blocks
parser.add_argument("--prompt-block-prob", type=float, default=0.05)
# Eval / Save
parser.add_argument("--eval-every", type=int, default=250, help="-1 = disable")
parser.add_argument("--eval-steps", type=int, default=20)
parser.add_argument("--save-every", type=int, default=-1, help="-1 = save only at end")
# Data
parser.add_argument("--data-dir", type=str, default="cache_nanochat/base_data_climbmix")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# DDP / device init
# ---------------------------------------------------------------------------
if "RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
master_process = ddp_rank == 0


def print0(*a, **kw):
    if master_process:
        print(*a, **kw, flush=True)


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
if args.run == "dummy" or not master_process:
    class _DummyWandb:
        def log(self, *a, **kw): pass
    wandb_run = _DummyWandb()
else:
    import wandb
    wandb_run = wandb.init(project="cola-pretrain", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load VAE (frozen) and build DiT from scratch
# ---------------------------------------------------------------------------
from cola_dlm import ColaTextVAEModel, ColaDiTModel
from cola_dlm.configuration_cola_dit import ColaDiTConfig
from cola_dlm.attention_utils import create_2l_block_causal_mask

print0("Loading VAE...")
vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)

print0("Initializing DiT from scratch...")
dit_config = ColaDiTConfig(
    txt_in_channels=vae.config.latent_dim,
    txt_out_channels=vae.config.latent_dim,
    txt_dim=args.dit_txt_dim,
    emb_dim=args.dit_txt_dim,
    heads=args.dit_heads,
    head_dim=args.dit_head_dim,
    expand_ratio=args.dit_expand_ratio,
    num_layers=args.dit_num_layers,
    patch_size=1,
    block_size=args.dit_block_size,
)
dit = ColaDiTModel(dit_config).to(device).train()

default_block_size = dit.block_size
latent_dim = vae.config.latent_dim
T = args.T

assert vae.patch_size == 1

# Block size config
n_candidates = int(math.log2(default_block_size)) + 1
BLOCK_SIZES = [2 ** i for i in range(n_candidates)]
if args.block_size_probs is not None:
    BLOCK_SIZE_PROBS = [float(x) for x in args.block_size_probs.split(",")]
else:
    BLOCK_SIZE_PROBS = [0.1] + [0.0] + [0.3] * (n_candidates - 2)
assert len(BLOCK_SIZE_PROBS) == n_candidates
assert abs(sum(BLOCK_SIZE_PROBS) - 1.0) < 1e-6

num_params = sum(p.numel() for p in dit.parameters())
print0(f"DiT: {num_params:,} params (random init), block_size={default_block_size}")
print0(f"Training block sizes: {dict(zip(BLOCK_SIZES, BLOCK_SIZE_PROBS))}")
print0(f"VAE: {sum(p.numel() for p in vae.parameters()):,} params (frozen)")

# ---------------------------------------------------------------------------
# Batch size and training horizon auto-computation
# ---------------------------------------------------------------------------
B_REF = 2 ** 19  # reference batch size in tokens (nanochat d12)
target_tokens = int(args.target_param_data_ratio * num_params)

if args.total_batch_size == -1:
    D_REF = args.target_param_data_ratio * B_REF
    predicted = B_REF * (target_tokens / D_REF) ** 0.383
    total_batch_tokens = 2 ** round(math.log2(predicted))
else:
    total_batch_tokens = args.total_batch_size

total_batch_sequences = total_batch_tokens // args.max_seq_len
world_tokens_per_fwd = args.device_batch_size * args.max_seq_len * ddp_world_size
grad_accum_steps = max(1, total_batch_tokens // world_tokens_per_fwd)
effective_batch_tokens = world_tokens_per_fwd * grad_accum_steps

if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    num_iterations = target_tokens // effective_batch_tokens
    print0(f"Auto training horizon: {target_tokens:,} tokens / {effective_batch_tokens:,} tokens/step = {num_iterations} iterations")

print0(f"Batch: {args.device_batch_size} x {grad_accum_steps} accum x {ddp_world_size} GPUs = {effective_batch_tokens:,} tokens/step")

# ---------------------------------------------------------------------------
# Optimizer: Muon for 2D, AdamW for 1D/scalar
# ---------------------------------------------------------------------------
from cola_dlm.optim import MuonAdamW, DistMuonAdamW

muon_groups = {}  # shape -> [params]
adamw_params = []

for name, p in dit.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 2:
        s = p.shape
        if s not in muon_groups:
            muon_groups[s] = []
        muon_groups[s].append(p)
    else:
        adamw_params.append(p)

param_groups = []
if adamw_params:
    param_groups.append({
        "params": adamw_params,
        "kind": "adamw",
        "lr": args.scalar_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "initial_lr": args.scalar_lr,
    })
for shape, params in muon_groups.items():
    param_groups.append({
        "params": params,
        "kind": "muon",
        "lr": args.matrix_lr,
        "momentum": 0.95,
        "ns_steps": 5,
        "beta2": 0.7,
        "weight_decay": args.weight_decay,
        "initial_lr": args.matrix_lr,
    })

if ddp_world_size > 1:
    optimizer = DistMuonAdamW(param_groups)
    print0("Optimizer: DistMuonAdamW (no DDP wrapper)")
else:
    optimizer = MuonAdamW(param_groups)
    print0("Optimizer: MuonAdamW (single GPU)")

# ---------------------------------------------------------------------------
# LR schedule: warmup + constant + warmdown (nanochat style)
# ---------------------------------------------------------------------------
def get_lr_multiplier(step):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
from cola_dlm.dataloader import pretrain_data_loader

STOP_TOKEN_ID = 47774

print0("Initializing data loader...")
data_dir = os.path.abspath(args.data_dir)
train_loader = pretrain_data_loader(
    args.tokenizer_path, data_dir, args.device_batch_size, args.max_seq_len,
    split="train", device=device,
)
val_loader = pretrain_data_loader(
    args.tokenizer_path, data_dir, args.device_batch_size, args.max_seq_len,
    split="val", device=device,
)

# ---------------------------------------------------------------------------
# Block size sampling
# ---------------------------------------------------------------------------
def sample_block_size():
    return BLOCK_SIZES[torch.multinomial(torch.tensor(BLOCK_SIZE_PROBS), 1).item()]


# ---------------------------------------------------------------------------
# Noisy copy construction
# ---------------------------------------------------------------------------
def build_noisy_sample_pretrain(z_0, t_val, z_1, sample_block_size):
    L = z_0.shape[0]
    z_noisy = (1 - t_val) * z_0 + t_val * z_1
    loss_mask = torch.ones(L, device=z_0.device)
    ts_noisy = torch.full((L,), t_val * T, device=z_0.device)
    target = z_1 - z_0

    if args.prompt_block_prob > 0:
        for blk_start in range(0, L, sample_block_size):
            blk_end = min(blk_start + sample_block_size, L)
            blk_len = blk_end - blk_start
            if blk_len <= 1:
                continue
            if random.random() < args.prompt_block_prob:
                split = random.randint(1, blk_len - 1)
                for j in range(split):
                    pos = blk_start + j
                    z_noisy[pos] = z_0[pos]
                    ts_noisy[pos] = 0.0
                    loss_mask[pos] = 0.0

    return z_noisy, loss_mask, target, ts_noisy


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------
def sample_timestep(batch_size):
    if args.timestep_dist == "uniform":
        return torch.rand(batch_size, device=device)
    u = torch.randn(batch_size, device=device)
    return torch.sigmoid(args.logit_normal_loc + args.logit_normal_scale * u)


# ---------------------------------------------------------------------------
# Prepare batch
# ---------------------------------------------------------------------------
def prepare_batch(inputs):
    B = inputs.shape[0]
    batch = []
    for i in range(B):
        bs = sample_block_size()
        token_row = inputs[i]
        L = token_row.shape[0]
        pad_len = (bs - L % bs) % bs
        if pad_len > 0:
            token_row = torch.cat([
                token_row,
                torch.full((pad_len,), STOP_TOKEN_ID, device=device, dtype=torch.long),
            ])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc = vae.encode([token_row])
        if args.vae_mode == "sample" and enc.latent_dists is not None:
            z_0 = enc.latent_dists[0].sample().float()
        else:
            z_0 = enc.latents_list[0].float()
        z_0 = (z_0 - vae.shifting_factor) * vae.scaling_factor
        batch.append((z_0, z_0.shape[0], bs))
    return batch


# ---------------------------------------------------------------------------
# Flow matching step
# ---------------------------------------------------------------------------
def flow_matching_step(dit_model, batch):
    B = len(batch)
    t = sample_timestep(B)

    extended_list = []
    target_list = []
    mask_list = []
    seq_lens = []
    block_sizes_list = []
    k_pos_list = []
    q_pos_list = []
    ts_list = []

    for i, (z_0, L, bs) in enumerate(batch):
        z_1 = torch.randn_like(z_0)
        z_noisy, loss_mask, target_vel, ts_noisy = build_noisy_sample_pretrain(
            z_0, t[i].item(), z_1, bs
        )

        extended_list.append(torch.cat([z_0.detach(), z_noisy], dim=0))
        target_list.append(target_vel)
        mask_list.append(loss_mask)
        seq_lens.append(L)
        block_sizes_list.append(bs)

        positions = torch.arange(L, device=device)
        k_pos_list.append(torch.cat([positions, positions]))
        q_pos_list.append(torch.cat([positions, positions]))
        ts_list.append(torch.cat([torch.zeros(L, device=device), ts_noisy]))

    txt = torch.cat(extended_list, dim=0)
    ext_lens = [2 * sl for sl in seq_lens]
    txt_shape = torch.tensor([[el] for el in ext_lens], dtype=torch.long, device=device)
    txt_q_shape = txt_shape.clone()

    k_position_ids = torch.cat(k_pos_list, dim=0)
    q_position_ids = torch.cat(q_pos_list, dim=0)
    timestep = torch.cat(ts_list, dim=0)

    attn_mask = create_2l_block_causal_mask(
        txt_shape, txt_q_shape,
        seq_lens=seq_lens, block_size=block_sizes_list,
        dtype=torch.bfloat16, device=device,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = dit_model(
            txt=txt.to(torch.bfloat16),
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            timestep=timestep.to(torch.bfloat16),
            k_position_ids=k_position_ids,
            q_position_ids=q_position_ids,
            attn_mask_override=attn_mask,
        )

    pred_list = []
    offset = 0
    for i, sl in enumerate(seq_lens):
        sample_out = out.txt_sample[offset : offset + 2 * sl]
        pred_list.append(sample_out[sl:])
        offset += 2 * sl

    pred = torch.cat(pred_list, dim=0).float()
    target = torch.cat(target_list, dim=0)
    loss_mask = torch.cat(mask_list, dim=0)

    error = ((pred - target) ** 2).mean(dim=-1)
    num_masked = loss_mask.sum().clamp(min=1.0)
    loss = (error * loss_mask).sum() / num_masked

    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(dit_model, eval_steps):
    dit_model.eval()
    losses = []
    for _ in range(eval_steps):
        inputs, _, _ = next(val_loader)
        batch = prepare_batch(inputs)
        loss = flow_matching_step(dit_model, batch)
        losses.append(loss.item())
    dit_model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(step, val_loss):
    if not master_process:
        return
    ckpt_dir = os.path.join(args.output_dir, args.run)
    os.makedirs(ckpt_dir, exist_ok=True)

    dit_path = os.path.join(ckpt_dir, f"dit_step_{step:06d}")
    dit.save_pretrained(dit_path)

    meta = {"step": step, "val_fm_loss": val_loss, "config": vars(args)}
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print0(f"Saved checkpoint at step {step} to {dit_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"LR schedule: warmup {args.warmup_steps} steps, warmdown {args.warmdown_ratio}, final {args.final_lr_frac}")
print0(f"Prompt-block prob: {args.prompt_block_prob}")
print0(f"Starting training for {num_iterations} iterations...")

smooth_loss = 0.0
ema_beta = 0.95
val_loss = float("nan")

for step in range(num_iterations):
    t0 = time.time()
    last_step = step == num_iterations - 1

    # --- Eval ---
    if step == 0 or last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        val_loss = evaluate(dit, args.eval_steps)
        print0(f"Step {step:06d} | Val FM loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "val/fm_loss": val_loss})

    # --- Save ---
    if last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0):
        save_checkpoint(step, val_loss)

    # --- Training step ---
    for micro_step in range(grad_accum_steps):
        inputs, _, _ = next(train_loader)
        batch = prepare_batch(inputs)
        loss = flow_matching_step(dit, batch)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()

    # LR update
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # Optimizer step (DistMuonAdamW handles gradient sync)
    torch.nn.utils.clip_grad_norm_(dit.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0

    # Logging
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))

    if step % 10 == 0 or last_step:
        print0(f"step {step:06d} | loss: {debiased:.6f} | lr_mul: {lrm:.4f} | dt: {dt * 1000:.0f}ms")

    wandb_run.log({
        "step": step,
        "train/loss": debiased,
        "train/raw_loss": train_loss.item(),
        "train/lr_multiplier": lrm,
        "train/dt": dt,
    })

# Cleanup
if ddp_world_size > 1:
    dist.destroy_process_group()

print0("Training complete.")
