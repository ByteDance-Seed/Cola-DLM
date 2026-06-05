"""
Supervised fine-tuning (SFT) for Cola-DLM.

Freezes the VAE and trains only the DiT (flow matching model) using the
2L trick for block-causal training.

Single-GPU:
    python scripts/cola_sft.py --num-iterations=100 --run=dummy

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=8 scripts/cola_sft.py --run=my_run
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cola-DLM SFT")
# Model
parser.add_argument("--dit-path", type=str, default="hf_models/cola_dlm/cola_dit")
parser.add_argument("--vae-path", type=str, default="hf_models/cola_dlm/cola_vae")
parser.add_argument("--tokenizer-path", type=str, default="hf_models/tokenizer.json")
# Output
parser.add_argument("--output-dir", type=str, default="cola_sft_checkpoints")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' = no logging)")
# Training
parser.add_argument("--num-iterations", type=int, default=-1, help="-1 = full epoch")
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--grad-accum-steps", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=512)
# Optimizer
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=0.01)
# Schedule
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.3)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Flow matching
parser.add_argument("--timestep-dist", type=str, default="logit_normal", choices=["logit_normal", "uniform"])
parser.add_argument("--logit-normal-loc", type=float, default=0.0)
parser.add_argument("--logit-normal-scale", type=float, default=1.0)
parser.add_argument("--T", type=float, default=1000.0)
# VAE
parser.add_argument("--vae-mode", type=str, default="sample", choices=["sample", "mode"])
# Data
parser.add_argument("--mmlu-epochs", type=int, default=3)
parser.add_argument("--gsm8k-epochs", type=int, default=4)
# Eval / Save
parser.add_argument("--eval-every", type=int, default=200, help="-1 = disable")
parser.add_argument("--eval-steps", type=int, default=20)
parser.add_argument("--save-every", type=int, default=-1, help="-1 = save only at end")
# Mode
parser.add_argument("--loss-mode", type=str, default="sft", choices=["sft", "all_token"])
# Chat format
parser.add_argument("--chat-format", type=str, default="text", choices=["text", "chatml"])
# Padding strategy
parser.add_argument("--pad-with-stop", type=str2bool, default=True,
                    help="Pad end-of-sequence with stop token (default: True)")
# Multi-turn boundary handling
parser.add_argument("--boundary-mode", type=str, default="token", choices=["token", "block"],
                    help="'token': per-position leakage prevention (handles all patterns). "
                         "'block': block-level [P,R]/[R,P] classification (legacy).")
# Block size randomization
parser.add_argument("--block-size-probs", type=str, default=None,
                    help="comma-separated probs for block sizes 1,2,4,...,block_size. "
                         "Length must be log2(block_size)+1. Default: 0.1,0,0.3,0.3,0.3")
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
        def log(self, *a, **kw):
            pass

    wandb_run = _DummyWandb()
else:
    import wandb

    wandb_run = wandb.init(project="cola-sft", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
from cola_dlm import ColaDiTModel, ColaTextVAEModel
from cola_dlm.attention_utils import create_2l_block_causal_mask
from tokenizers import Tokenizer

print0("Loading models...")
vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)

dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).train()

tokenizer = Tokenizer.from_file(args.tokenizer_path)

default_block_size = dit.block_size
latent_dim = vae.config.latent_dim
patch_size = vae.patch_size
T = args.T

# patch_size > 1 not supported: role-to-latent mapping, boundary detection,
# and token trimming all assume 1:1 token-to-latent correspondence.
assert patch_size == 1, f"Only patch_size=1 is supported, got {patch_size}"

# Block size config: candidates are 1, 2, 4, ..., default_block_size
n_candidates = int(math.log2(default_block_size)) + 1
BLOCK_SIZES = [2 ** i for i in range(n_candidates)]  # [1, 2, 4, 8, 16]
if args.block_size_probs is not None:
    BLOCK_SIZE_PROBS = [float(x) for x in args.block_size_probs.split(",")]
else:
    # Default: 0.1, 0, 0.3, 0.3, 0.3 (skip block_size=2)
    BLOCK_SIZE_PROBS = [0.1] + [0.0] + [0.3] * (n_candidates - 2)
assert len(BLOCK_SIZE_PROBS) == n_candidates, (
    f"block-size-probs must have {n_candidates} values for block sizes {BLOCK_SIZES}, "
    f"got {len(BLOCK_SIZE_PROBS)}"
)
assert abs(sum(BLOCK_SIZE_PROBS) - 1.0) < 1e-6, f"block-size-probs must sum to 1, got {sum(BLOCK_SIZE_PROBS)}"

print0(f"DiT: {sum(p.numel() for p in dit.parameters()):,} params, default_block_size={default_block_size}")
print0(f"Training block sizes: {dict(zip(BLOCK_SIZES, BLOCK_SIZE_PROBS))}")
print0(f"VAE: {sum(p.numel() for p in vae.parameters()):,} params (frozen), latent_dim={latent_dim}")

# DDP wrap DiT only
orig_dit = dit
if ddp_world_size > 1:
    dit = torch.nn.parallel.DistributedDataParallel(dit, device_ids=[ddp_local_rank])

# ---------------------------------------------------------------------------
# Optimizer + LR schedule
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    (orig_dit if ddp_world_size > 1 else dit).parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]


def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac


# ---------------------------------------------------------------------------
# Conversation rendering
# ---------------------------------------------------------------------------
IM_START = 100264
IM_END = 100265
PAD_TOKEN = 100277  # <|pad|> — needs finetuned VAE for chatml mode
STOP_TOKEN_ID = 47774  # ■ — single token, VAE-reconstructable, no clash with data


def render_conversation(conversation, max_tokens, chat_format="text"):
    """Render a conversation dict into token IDs + roles.

    Returns (ids: list[int], roles: list[str]) where roles[i] is 'P' or 'R'.
    """
    ids, roles = [], []

    def add(token_ids, role):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        roles.extend([role] * len(token_ids))

    messages = list(conversation["messages"])
    if messages and messages[0]["role"] == "system":
        messages[1] = {
            "role": messages[1]["role"],
            "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
        }
        messages = messages[1:]

    if chat_format == "text":
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(p["text"] for p in content)
            if message["role"] in ("user", "system"):
                add(tokenizer.encode(f"User: {content}\n").ids, "P")
            elif message["role"] == "assistant":
                add(tokenizer.encode("Assistant: ").ids, "P")
                add(tokenizer.encode(f"{content}").ids, "R")
                add(STOP_TOKEN_ID, "R")  # ■
                add(tokenizer.encode("\n").ids, "R")
    elif chat_format == "chatml":
        for message in messages:
            content = message["content"]
            header_ids = tokenizer.encode(f"{message['role']}\n").ids
            add(IM_START, "P")
            add(header_ids, "P")
            if message["role"] == "assistant":
                if isinstance(content, str):
                    add(tokenizer.encode(content).ids, "R")
                elif isinstance(content, list):
                    for part in content:
                        r = "R" if part.get("type") in ("text", "python") else "P"
                        add(tokenizer.encode(part["text"]).ids, r)
                add(IM_END, "R")
            else:
                if isinstance(content, str):
                    add(tokenizer.encode(content).ids, "P")
                add(IM_END, "P")
            add(tokenizer.encode("\n").ids, "P")

    return ids[:max_tokens], roles[:max_tokens]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee

print0("Loading datasets...")
train_dataset = TaskMixture(
    [
        SmolTalk(split="train"),
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]
)
val_dataset = TaskMixture(
    [
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ]
)
print0(f"Train: {len(train_dataset):,} conversations, Val: {len(val_dataset):,}")


# ---------------------------------------------------------------------------
# Noisy copy construction (boundary-aware)
# ---------------------------------------------------------------------------
def build_noisy_sample(z_0, roles_str, t_val, z_1, loss_mode, sample_block_size, boundary_mode):
    """Construct the noisy copy with boundary-aware noising.

    Returns:
        z_noisy: (L, d)
        loss_mask: (L,)
        target: (L, d) velocity target = z_1 - z_0
        ts_noisy: (L,) per-position timestep (0 for clean, t*T for noised, T for masked)
    """
    L = z_0.shape[0]
    z_noisy = z_0.clone()
    loss_mask = torch.zeros(L, device=z_0.device)
    ts_noisy = torch.zeros(L, device=z_0.device)
    target = z_1 - z_0

    for blk_start in range(0, L, sample_block_size):
        blk_end = min(blk_start + sample_block_size, L)
        blk_roles = roles_str[blk_start:blk_end]

        if boundary_mode == "token":
            # Per-position: any P after an R in the same block is masked.
            # Handles [P,R], [R,P], [P,R,P], [R,P,R], etc.
            seen_r = False
            for j in range(blk_end - blk_start):
                pos = blk_start + j
                if blk_roles[j] == "R":
                    seen_r = True
                    z_noisy[pos] = (1 - t_val) * z_0[pos] + t_val * z_1[pos]
                    ts_noisy[pos] = t_val * T
                    loss_mask[pos] = 1.0
                elif seen_r:
                    # P after R → mask with pure noise to prevent leakage
                    z_noisy[pos] = z_1[pos]
                    ts_noisy[pos] = t_val * T
                    loss_mask[pos] = 0.0
                else:
                    # P before any R → safe conditioning context
                    if loss_mode == "all_token":
                        z_noisy[pos] = (1 - t_val) * z_0[pos] + t_val * z_1[pos]
                        ts_noisy[pos] = t_val * T
                        loss_mask[pos] = 1.0
                    else:
                        z_noisy[pos] = z_0[pos]
                        ts_noisy[pos] = 0.0
                        loss_mask[pos] = 0.0

        else:  # boundary_mode == "block"
            has_p = "P" in blk_roles
            has_r = "R" in blk_roles

            if has_p and has_r:
                first_p = next(i for i, r in enumerate(blk_roles) if r == "P")
                first_r = next(i for i, r in enumerate(blk_roles) if r == "R")
                rp_boundary = first_r < first_p

            for j in range(blk_end - blk_start):
                pos = blk_start + j
                is_r = blk_roles[j] == "R"

                if has_p and has_r:
                    if is_r:
                        z_noisy[pos] = (1 - t_val) * z_0[pos] + t_val * z_1[pos]
                        ts_noisy[pos] = t_val * T
                        loss_mask[pos] = 1.0
                    elif rp_boundary:
                        z_noisy[pos] = z_1[pos]
                        ts_noisy[pos] = t_val * T
                        loss_mask[pos] = 0.0
                    else:
                        if loss_mode == "all_token":
                            z_noisy[pos] = (1 - t_val) * z_0[pos] + t_val * z_1[pos]
                            ts_noisy[pos] = t_val * T
                            loss_mask[pos] = 1.0
                        else:
                            z_noisy[pos] = z_0[pos]
                            ts_noisy[pos] = 0.0
                            loss_mask[pos] = 0.0
                elif has_r:
                    z_noisy[pos] = (1 - t_val) * z_0[pos] + t_val * z_1[pos]
                    ts_noisy[pos] = t_val * T
                    loss_mask[pos] = 1.0
                else:
                    if loss_mode == "all_token":
                        z_noisy[pos] = (1 - t_val) * z_0[pos] + t_val * z_1[pos]
                        ts_noisy[pos] = t_val * T
                        loss_mask[pos] = 1.0
                    else:
                        z_noisy[pos] = z_0[pos]
                        ts_noisy[pos] = 0.0
                        loss_mask[pos] = 0.0

    return z_noisy, loss_mask, target, ts_noisy


# ---------------------------------------------------------------------------
# Sample preparation
# ---------------------------------------------------------------------------
def sample_block_size():
    """Sample a block size from the configured distribution."""
    return BLOCK_SIZES[torch.multinomial(torch.tensor(BLOCK_SIZE_PROBS), 1).item()]


def prepare_sample(conversation, max_seq_len, vae_mode, sample_bs=None):
    """Tokenize, encode through VAE.

    Returns None if the conversation is too short or has no response.
    """
    if sample_bs is None:
        sample_bs = sample_block_size()

    ids, roles_str = render_conversation(conversation, max_seq_len, args.chat_format)

    if "R" not in roles_str:
        return None

    # Pad end of sequence
    if args.chat_format == "chatml":
        pad_token = IM_END if args.pad_with_stop else PAD_TOKEN
    else:
        pad_token = STOP_TOKEN_ID
    pad_len = (sample_bs - len(ids) % sample_bs) % sample_bs
    ids = ids + [pad_token] * pad_len
    roles_str = roles_str + ["R"] * pad_len

    L = len(ids)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = vae.encode([input_ids])

    if vae_mode == "sample" and enc.latent_dists is not None:
        z_0 = enc.latent_dists[0].sample().float()
    else:
        z_0 = enc.latents_list[0].float()

    z_0 = (z_0 - vae.shifting_factor) * vae.scaling_factor

    return z_0, roles_str, L, sample_bs


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------
def sample_timestep(batch_size):
    if args.timestep_dist == "uniform":
        return torch.rand(batch_size, device=device)
    else:
        u = torch.randn(batch_size, device=device)
        return torch.sigmoid(args.logit_normal_loc + args.logit_normal_scale * u)


# ---------------------------------------------------------------------------
# Flow matching step
# ---------------------------------------------------------------------------
def flow_matching_step(dit_model, batch):
    """One FM training step on a batch.

    batch: list of (z_0, roles_str, L, sample_bs) tuples from prepare_sample.
    Returns: scalar loss.
    """
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

    for i, (z_0, roles_str, L, sample_bs) in enumerate(batch):
        z_1 = torch.randn_like(z_0)
        z_noisy, loss_mask, target_vel, ts_noisy = build_noisy_sample(
            z_0, roles_str, t[i].item(), z_1, args.loss_mode, sample_bs, args.boundary_mode
        )

        extended_list.append(torch.cat([z_0.detach(), z_noisy], dim=0))
        target_list.append(target_vel)
        mask_list.append(loss_mask)
        seq_lens.append(L)
        block_sizes_list.append(sample_bs)

        positions = torch.arange(L, device=device)
        k_pos_list.append(torch.cat([positions, positions]))
        q_pos_list.append(torch.cat([positions, positions]))

        # Per-position timestep: clean copy=0, noisy copy=ts_noisy
        ts_list.append(torch.cat([torch.zeros(L, device=device), ts_noisy]))

    # NA-form concatenation
    txt = torch.cat(extended_list, dim=0)
    ext_lens = [2 * sl for sl in seq_lens]
    txt_shape = torch.tensor([[el] for el in ext_lens], dtype=torch.long, device=device)
    txt_q_shape = txt_shape.clone()

    k_position_ids = torch.cat(k_pos_list, dim=0)
    q_position_ids = torch.cat(q_pos_list, dim=0)
    timestep = torch.cat(ts_list, dim=0)

    # 2L attention mask (per-sample block sizes)
    attn_mask = create_2l_block_causal_mask(
        txt_shape,
        txt_q_shape,
        seq_lens=seq_lens,
        block_size=block_sizes_list,
        dtype=torch.bfloat16,
        device=device,
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
# Data generator
# ---------------------------------------------------------------------------
def data_generator(dataset, batch_size, max_seq_len, vae_mode):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    rng = random.Random(42 + ddp_rank)
    epoch = 0

    while True:
        rng.shuffle(indices)
        cursor = 0
        while cursor < dataset_size:
            batch = []
            while len(batch) < batch_size and cursor < dataset_size:
                idx = indices[cursor]
                cursor += 1
                if (cursor - 1) % ddp_world_size != ddp_rank:
                    continue
                sample = prepare_sample(dataset[idx], max_seq_len, vae_mode)
                if sample is not None:
                    batch.append(sample)
            if len(batch) == batch_size:
                yield batch, epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(dit_model, val_gen, eval_steps):
    dit_model.eval()
    losses = []
    for _ in range(eval_steps):
        batch, _ = next(val_gen)
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
    # Save SFT config into the model config for inference auto-detection
    dit_save.config.sft_chat_format = args.chat_format
    dit_save.config.sft_pad_with_stop = args.pad_with_stop
    dit_save.config.sft_stop_token_id = IM_END if args.chat_format == "chatml" else STOP_TOKEN_ID
    dit_save.save_pretrained(dit_path)

    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, f"optim_{step:06d}.pt"))

    meta = {
        "step": step,
        "val_fm_loss": val_loss,
        "config": vars(args),
    }
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print0(f"Saved checkpoint at step {step} to {dit_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"Loss mode: {args.loss_mode}")
print0(f"Timestep dist: {args.timestep_dist}")
print0(f"Batch: {args.device_batch_size} x {args.grad_accum_steps} accum = {args.device_batch_size * args.grad_accum_steps} effective")
print0(f"LR: {args.learning_rate}, warmup: {args.warmup_ratio}, warmdown: {args.warmdown_ratio}")

train_gen = data_generator(train_dataset, args.device_batch_size, args.max_seq_len, args.vae_mode)
val_gen = data_generator(val_dataset, args.device_batch_size, args.max_seq_len, "mode")

# Determine total iterations
if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    approx_samples = len(train_dataset) // ddp_world_size
    num_iterations = approx_samples // (args.device_batch_size * args.grad_accum_steps)
    print0(f"Full epoch: ~{num_iterations} iterations")

smooth_loss = 0.0
ema_beta = 0.95
val_loss = float("nan")

print0(f"Starting training for {num_iterations} iterations...")

for step in range(num_iterations):
    t0 = time.time()
    last_step = step == num_iterations - 1

    # --- Eval ---
    if step == 0 or last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        val_loss = evaluate(dit, val_gen, args.eval_steps)
        print0(f"Step {step:05d} | Val FM loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "val/fm_loss": val_loss})

    # --- Save ---
    if last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0):
        save_checkpoint(step, val_loss)

    # --- Training step ---
    for micro_step in range(args.grad_accum_steps):
        is_last_micro = micro_step == args.grad_accum_steps - 1
        ctx = nullcontext() if (is_last_micro or ddp_world_size == 1) else dit.no_sync()
        with ctx:
            batch, epoch = next(train_gen)
            loss = flow_matching_step(dit, batch)
            train_loss = loss.detach()
            (loss / args.grad_accum_steps).backward()

    # LR update
    progress = step / max(num_iterations - 1, 1)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(
        (orig_dit if ddp_world_size > 1 else dit).parameters(), 1.0
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0

    # Logging
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))

    if step % 10 == 0 or last_step:
        print0(
            f"step {step:05d} | loss: {debiased:.6f} | "
            f"lr: {lrm * args.learning_rate:.2e} | dt: {dt * 1000:.0f}ms | epoch: {epoch}"
        )

    wandb_run.log(
        {
            "step": step,
            "train/loss": debiased,
            "train/raw_loss": train_loss.item(),
            "train/lr": lrm * args.learning_rate,
            "train/dt": dt,
            "train/epoch": epoch,
        }
    )

# Cleanup
if ddp_world_size > 1:
    dist.destroy_process_group()

print0("Training complete.")
