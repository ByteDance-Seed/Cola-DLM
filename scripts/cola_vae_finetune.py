"""
Quick VAE embedding finetuning to learn special chat tokens.

Unfreezes only the token embedding (encoder.wte) and output projection
(decoder.final_layer) while keeping all transformer blocks frozen.
Trains with reconstruction loss on chat-formatted data.

Usage:
    python scripts/cola_vae_finetune.py --num-iterations=500 --run=dummy
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description="Cola-DLM VAE embedding finetuning")
parser.add_argument("--vae-path", type=str, default="hf_models/cola_dlm/cola_vae")
parser.add_argument("--tokenizer-path", type=str, default="hf_models/tokenizer.json")
parser.add_argument("--output-dir", type=str, default="cola_sft_checkpoints")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--num-iterations", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--eval-every", type=int, default=50)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from cola_dlm import ColaTextVAEModel
from cola_dlm.attention_utils import create_na_block_causal_mask

print("Loading VAE...")
vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device)
tokenizer = Tokenizer.from_file(args.tokenizer_path)

# Freeze everything
for p in vae.parameters():
    p.requires_grad_(False)

# Unfreeze only decoder output projection (preserves latent space)
for p in vae.decoder.final_layer.parameters():
    p.requires_grad_(True)

total_params = sum(p.numel() for p in vae.parameters())
trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print(f"VAE: {total_params:,} total, {trainable_params:,} trainable ({100*trainable_params/total_params:.1f}%)")

# Special tokens to verify
IM_START = 100264
IM_END = 100265
SPECIAL_TOKENS = {IM_START: "<|im_start|>", IM_END: "<|im_end|>"}

# Build training data: ChatML-formatted conversations
# Mix of simple chat patterns to teach the VAE the special tokens
CHAT_TEMPLATES = [
    "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n",
]

QUESTIONS = [
    "Hello!", "How are you?", "What is 2+2?", "Tell me a joke.",
    "What is the capital of France?", "Explain gravity.", "Hi there!",
    "What is Python?", "Who are you?", "Help me write code.",
    "What time is it?", "Good morning!", "Can you help?",
    "What is AI?", "Translate hello to Spanish.", "How does rain form?",
]

ANSWERS = [
    "Hi! How can I help?", "I'm doing well, thanks!", "2+2 equals 4.",
    "Why did the chicken cross the road? To get to the other side!",
    "The capital of France is Paris.", "Gravity is a force of attraction.",
    "Hello! What can I do for you?", "Python is a programming language.",
    "I'm an AI assistant.", "Of course! What do you need?",
    "I don't have access to the current time.", "Good morning!",
    "Sure, I'd be happy to help!", "AI stands for artificial intelligence.",
    "Hello in Spanish is 'hola'.", "Rain forms when water evaporates.",
]

block_size = vae.block_size
patch_size = vae.patch_size
chunk = block_size * patch_size


def make_batch(batch_size, max_seq_len):
    """Create a batch of ChatML-formatted sequences."""
    import random
    input_ids_list = []
    for _ in range(batch_size):
        template = random.choice(CHAT_TEMPLATES)
        q = random.choice(QUESTIONS)
        a = random.choice(ANSWERS)
        text = template.format(q=q, a=a)
        ids = tokenizer.encode(text).ids[:max_seq_len]
        pad_len = (chunk - len(ids) % chunk) % chunk
        ids = ids + [100277] * pad_len  # pad token
        input_ids_list.append(torch.tensor(ids, dtype=torch.long, device=device))
    return input_ids_list


def compute_loss(vae, input_ids_list):
    """Reconstruction loss: encode → decode → cross-entropy."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc = vae.encode(input_ids_list)

    # Use mode (deterministic) for stability
    z_list = enc.latents_list

    # Decode each sample
    total_loss = 0.0
    total_tokens = 0
    for i, (z, input_ids) in enumerate(zip(z_list, input_ids_list)):
        txt_shape = torch.tensor([[z.shape[0]]], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = vae.decode(z=z, txt_shape=txt_shape, txt_q_shape=txt_shape)
        logits = logits.view(-1, logits.shape[-1]).float()
        targets = input_ids[:logits.shape[0]]
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss
        total_tokens += targets.shape[0]

    return total_loss / total_tokens


def eval_special_tokens(vae):
    """Check if the VAE can reconstruct special tokens in context."""
    text = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n"
    ids = tokenizer.encode(text).ids
    pad_len = (chunk - len(ids) % chunk) % chunk
    ids = ids + [100277] * pad_len

    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = vae.encode([input_ids])
        z = enc.latents_list[0]
        txt_shape = torch.tensor([[z.shape[0]]], device=device)
        logits = vae.decode(z=z, txt_shape=txt_shape, txt_q_shape=txt_shape)
        pred_ids = logits.view(-1, logits.shape[-1]).argmax(dim=-1).tolist()

    n_correct = 0
    n_special_correct = 0
    n_special_total = 0
    n_total = len(ids) - pad_len

    for i in range(n_total):
        if ids[i] == pred_ids[i]:
            n_correct += 1
        if ids[i] in SPECIAL_TOKENS:
            n_special_total += 1
            if ids[i] == pred_ids[i]:
                n_special_correct += 1
            else:
                pass  # will print below

    return n_correct, n_total, n_special_correct, n_special_total


# Optimizer
optimizer = torch.optim.AdamW(
    [p for p in vae.parameters() if p.requires_grad],
    lr=args.learning_rate,
)

# Training loop
print(f"Training for {args.num_iterations} iterations...")
vae.train()

for step in range(args.num_iterations):
    t0 = time.time()

    batch = make_batch(args.batch_size, args.max_seq_len)
    loss = compute_loss(vae, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0

    if step % 10 == 0:
        print(f"step {step:04d} | loss: {loss.item():.4f} | dt: {dt*1000:.0f}ms")

    if step == 0 or (step + 1) % args.eval_every == 0:
        vae.eval()
        correct, total, sp_correct, sp_total = eval_special_tokens(vae)
        print(f"  Eval: {correct}/{total} tokens correct ({100*correct/total:.1f}%), "
              f"special: {sp_correct}/{sp_total}")
        vae.train()

# Final eval
vae.eval()
correct, total, sp_correct, sp_total = eval_special_tokens(vae)
print(f"\nFinal: {correct}/{total} ({100*correct/total:.1f}%), special: {sp_correct}/{sp_total}")

# Save
if args.run != "dummy":
    save_path = os.path.join(args.output_dir, f"vae_{args.run}")
    vae.save_pretrained(save_path)
    print(f"Saved to {save_path}")
else:
    print("Dummy run, not saving.")
