"""
Continue pretraining Cola-DLM DiT on ClimbMix-400B data.

Freezes the VAE and trains only the DiT using flow matching with the 2L trick.
Hyperparameters default to the paper's Table 8 (AdamW, cosine LR schedule).

Single-GPU:
    python scripts/cola_cpt.py --num-iterations=100 --run=dummy

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=8 scripts/cola_cpt.py --run=my_run
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
parser = argparse.ArgumentParser(description="Cola-DLM Continue Pretraining")
# Model
parser.add_argument("--dit-path", type=str, default="hf_models/cola_dlm/cola_dit")
parser.add_argument("--vae-path", type=str, default="hf_models/cola_dlm/cola_vae")
parser.add_argument("--tokenizer-path", type=str, default="hf_models/tokenizer.json")
# Output
parser.add_argument("--output-dir", type=str, default="cola_cpt_checkpoints")
parser.add_argument("--run", type=str, default="dummy")
# Training (Table 8 defaults)
parser.add_argument("--num-iterations", type=int, default=1000000)
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--global-batch-size", type=int, default=1408)
parser.add_argument("--max-seq-len", type=int, default=512)
# Optimizer (Table 8)
parser.add_argument("--learning-rate", type=float, default=1.5e-4)
parser.add_argument("--initial-lr", type=float, default=1e-6)
parser.add_argument("--final-lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--warmup-steps", type=int, default=5000)
parser.add_argument("--grad-clip", type=float, default=1.0)
# Flow matching
parser.add_argument("--timestep-dist", type=str, default="logit_normal", choices=["logit_normal", "uniform"])
parser.add_argument("--logit-normal-loc", type=float, default=0.0)
parser.add_argument("--logit-normal-scale", type=float, default=1.0)
parser.add_argument("--T", type=float, default=1000.0)
# VAE
parser.add_argument("--vae-mode", type=str, default="sample", choices=["sample", "mode"])
# Block size randomization
parser.add_argument("--block-size-probs", type=str, default=None,
                    help="Comma-separated probs for block sizes 1,2,4,...,block_size")
# Simulated prompt-response blocks
parser.add_argument("--prompt-block-prob", type=float, default=0.05,
                    help="Prob of simulating a [P,R] boundary within a noisy block")
# Eval / Save
parser.add_argument("--eval-every", type=int, default=500, help="-1 = disable")
parser.add_argument("--eval-steps", type=int, default=20)
parser.add_argument("--save-every", type=int, default=10000, help="-1 = save only at end")
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
    wandb_run = wandb.init(project="cola-cpt", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
from cola_dlm import ColaDiTModel, ColaTextVAEModel
from cola_dlm.attention_utils import create_2l_block_causal_mask

print0("Loading models...")
vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)

dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).train()

default_block_size = dit.block_size
latent_dim = vae.config.latent_dim
patch_size = vae.patch_size
T = args.T

assert patch_size == 1, f"Only patch_size=1 is supported, got {patch_size}"

# Block size config
n_candidates = int(math.log2(default_block_size)) + 1
BLOCK_SIZES = [2 ** i for i in range(n_candidates)]
if args.block_size_probs is not None:
    BLOCK_SIZE_PROBS = [float(x) for x in args.block_size_probs.split(",")]
else:
    BLOCK_SIZE_PROBS = [0.1] + [0.0] + [0.3] * (n_candidates - 2)
assert len(BLOCK_SIZE_PROBS) == n_candidates
assert abs(sum(BLOCK_SIZE_PROBS) - 1.0) < 1e-6

print0(f"DiT: {sum(p.numel() for p in dit.parameters()):,} params, block_size={default_block_size}")
print0(f"Training block sizes: {dict(zip(BLOCK_SIZES, BLOCK_SIZE_PROBS))}")
print0(f"VAE: {sum(p.numel() for p in vae.parameters()):,} params (frozen)")

# DDP wrap
orig_dit = dit
if ddp_world_size > 1:
    dit = torch.nn.parallel.DistributedDataParallel(dit, device_ids=[ddp_local_rank])

# ---------------------------------------------------------------------------
# Optimizer (Table 8: AdamW, betas=(0.9, 0.95))
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    (orig_dit if ddp_world_size > 1 else dit).parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay,
)

# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay (Table 8)
# ---------------------------------------------------------------------------
def get_lr(step):
    if step < args.warmup_steps:
        return args.initial_lr + (args.learning_rate - args.initial_lr) * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.num_iterations - args.warmup_steps, 1)
    return args.final_lr + 0.5 * (args.learning_rate - args.final_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
from cola_dlm.dataloader import pretrain_data_loader

STOP_TOKEN_ID = 47774  # ■

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

# Grad accumulation
grad_accum_steps = max(1, args.global_batch_size // (args.device_batch_size * ddp_world_size))
effective_batch = args.device_batch_size * grad_accum_steps * ddp_world_size
print0(f"Batch: {args.device_batch_size} x {grad_accum_steps} accum x {ddp_world_size} GPUs = {effective_batch} effective")

# ---------------------------------------------------------------------------
# Block size sampling
# ---------------------------------------------------------------------------
def sample_block_size():
    return BLOCK_SIZES[torch.multinomial(torch.tensor(BLOCK_SIZE_PROBS), 1).item()]


# ---------------------------------------------------------------------------
# Noisy copy construction for pretraining
# ---------------------------------------------------------------------------
def build_noisy_sample_pretrain(z_0, t_val, z_1, sample_block_size):
    """All-token noisy construction with optional simulated [P,R] boundaries.

    With probability prompt_block_prob per block, a random split creates a
    clean left segment (ts=0, loss=0) and noisy right segment (ts=t*T, loss=1),
    simulating the conditioning pattern used at inference.
    """
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
                # Simulate [P, R] boundary: random split point
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
# Prepare batch: token IDs -> VAE latents
# ---------------------------------------------------------------------------
def prepare_batch(inputs):
    """Convert (B, T) token IDs to list of (z_0, L, block_size) tuples."""
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

    dit_save = orig_dit if ddp_world_size > 1 else dit
    dit_path = os.path.join(ckpt_dir, f"dit_step_{step:06d}")
    dit_save.save_pretrained(dit_path)

    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, f"optim_{step:06d}.pt"))

    meta = {"step": step, "val_fm_loss": val_loss, "config": vars(args)}
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print0(f"Saved checkpoint at step {step} to {dit_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"Optimizer: AdamW, betas=(0.9, 0.95), wd={args.weight_decay}")
print0(f"LR: warmup {args.initial_lr} -> {args.learning_rate} over {args.warmup_steps} steps, cosine -> {args.final_lr}")
print0(f"Prompt-block prob: {args.prompt_block_prob}")
print0(f"Starting training for {args.num_iterations} iterations...")

smooth_loss = 0.0
ema_beta = 0.95
val_loss = float("nan")

for step in range(args.num_iterations):
    t0 = time.time()
    last_step = step == args.num_iterations - 1

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
        is_last_micro = micro_step == grad_accum_steps - 1
        ctx = nullcontext() if (is_last_micro or ddp_world_size == 1) else dit.no_sync()
        with ctx:
            inputs, _, _ = next(train_loader)
            batch = prepare_batch(inputs)
            loss = flow_matching_step(dit, batch)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()

    # LR update
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(
        (orig_dit if ddp_world_size > 1 else dit).parameters(), args.grad_clip
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0

    # Logging
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))

    if step % 10 == 0 or last_step:
        print0(f"step {step:06d} | loss: {debiased:.6f} | lr: {lr:.2e} | dt: {dt * 1000:.0f}ms")

    wandb_run.log({
        "step": step,
        "train/loss": debiased,
        "train/raw_loss": train_loss.item(),
        "train/lr": lr,
        "train/dt": dt,
    })

# Cleanup
if ddp_world_size > 1:
    dist.destroy_process_group()

print0("Training complete.")
