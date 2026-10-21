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
Cola DLM inference script — DiT prior transport + VAE encode/decode (NA form).

This module is the open-source implementation of the three-step
inference algorithm of *Continuous Latent Diffusion Language Model*
(arXiv:2605.06548):

1. **Prefix encode** (Eq. 2.2.4)::

       z^pre ~ q_phi(z^pre | x^pre)

   computed by ``ColaTextVAEModel.encode``.

2. **Block-wise latent prior transport** (Eq. 2.2.5), realized by
   :func:`generate_task_repaint_inference` as a loop over generation
   blocks ``b = 1, 2, ..., B``. For each block we draw
   ``eps^(b) ~ N(0, I)`` and integrate the learned vector field
   ``v_psi`` of ``ColaDiTModel`` from ``t = T`` down to ``t = 0`` under
   the historical condition ``(z^pre, hat z_0^(<b))``::

       hat z_0^(b) = Phi^psi_{0<-1}(eps^(b); z^pre, hat z_0^(<b)).

3. **Conditional decode** (Eq. 2.2.6)::

       hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))

   computed by ``ColaTextVAEModel.decode`` one freshly transported
   block at a time.

The pipeline runs in NA / flatten-concat form: no batch-wise padding —
each sample contributes only its own per-sample block-aligned
``input_ids``, and per-sample lengths are carried in companion
``txt_shape`` / ``txt_q_shape`` tensors. Practical implementation
details on top of the paper algorithm — block-aligned prompt padding,
KV cache contracts, and the per-sample CFG fallback for prompts shorter
than ``block_size`` — are documented in-line below.

Usage::

    python -m cola_dlm.inference \
        --dit_path hf_models/cola_dlm/cola_dit \
        --vae_path hf_models/cola_dlm/cola_vae \
        --tokenizer_path hf_models/tokenizer.json \
        --input_jsonl generate_task_data/lambada.jsonl \
        --output_dir eval_output/my_run \
        --task_name lambada
"""

import argparse
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .modeling_cola_dit import ColaDiTModel
from .modeling_cola_vae import ColaTextVAEModel

# ---------------------------------------------------------------------------
# Prompt Templates (standalone, from prompt_template.py)
# ---------------------------------------------------------------------------


def apply_prompt_template(task: str, context: str, question: str, answer: str, choices: Optional[list[str]]) -> str:
    if task == "lambada":
        return question
    elif task == "squad":
        return (
            "Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway.\n"
            "Question: In what country is Normandy located?\n"
            "Answer: France\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    elif task == "obqa":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Question: Which tool is best for tightening a screw?\n"
            "(A) spoon\n"
            "(B) hammer\n"
            "(C) screwdriver\n"
            "(D) paintbrush\n"
            "Answer: screwdriver\n\n"
            "Question: What do plants absorb from the air during photosynthesis?\n"
            "(A) carbon dioxide\n"
            "(B) oxygen\n"
            "(C) helium\n"
            "(D) salt\n"
            "Answer: carbon dioxide\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    elif task == "hellaswag":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Context: The girl puts the bread into the toaster and pushes the lever down. The bread\n"
            "(A) becomes a slice of pizza.\n"
            "(B) starts to toast and turn brown.\n"
            "(C) disappears immediately.\n"
            "(D) turns into a glass of water.\n"
            "Answer: starts to toast and turn brown.\n\n"
            "Context: The goalkeeper sees the ball coming towards the net. He dives and\n"
            "(A) catches the ball with his hands.\n"
            "(B) starts dancing in the field.\n"
            "(C) opens a laptop to check email.\n"
            "(D) runs away from the stadium.\n"
            "Answer: catches the ball with his hands.\n\n"
            f"Context: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    elif task == "mmlu":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        return (
            "Question: Which gas do plants absorb from the air during photosynthesis?\n"
            "(A) Oxygen\n"
            "(B) Carbon dioxide\n"
            "(C) Nitrogen\n"
            "(D) Hydrogen\n"
            "Answer: Carbon dioxide\n\n"
            "Question: A triangle has angles 50 degrees and 60 degrees. What is the third angle?\n"
            "(A) 60 degrees\n"
            "(B) 70 degrees\n"
            "(C) 80 degrees\n"
            "(D) 90 degrees\n"
            "Answer: 70 degrees\n\n"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    elif task == "race":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        # Put the passage under "Article:" and, when the question is supplied as
        # its own field, label it under "Question:". Datasets that inline the
        # question into the passage keep everything under "Article:".
        if context:
            current_item = f"Article: {context}\nQuestion: {question}\n"
        else:
            current_item = f"Article: {question}\n"
        return (
            "Read the following article and answer the question.\n\n"
            "Article: Mary went to the store to buy some fruits. She bought five apples and two oranges. She paid 5 dollars in total.\n"
            "Question: What did Mary buy?\n"
            "Options:\n"
            "(A) Bananas\n"
            "(B) Apples and oranges\n"
            "(C) Grapes\n"
            "(D) Watermelon\n"
            "Answer: Apples and oranges\n\n"
            f"{current_item}"
            f"Options:\n{current_choices_text}\n"
            "Answer:"
        )
    elif task == "siqa":
        choices = choices or []
        current_choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        # Social IQa questions are under-determined without their situational
        # context, so surface it on a dedicated "Context:" line. Datasets that
        # inline the context into the question leave this field empty.
        context_block = f"Context: {context}\n" if context else ""
        return (
            "Context: Jordan wanted to tell a joke to his friends.\n"
            "Question: What does Jordan need to do before this?\n"
            "(A) ignore his friends\n"
            "(B) think of a funny story\n"
            "(C) leave the room\n"
            "Answer: think of a funny story\n\n"
            "Context: Kai helped his neighbor carry heavy groceries inside.\n"
            "Question: How would the neighbor feel?\n"
            "(A) angry\n"
            "(B) grateful\n"
            "(C) scared\n"
            "Answer: grateful\n\n"
            f"{context_block}"
            f"Question: {question}\n"
            f"{current_choices_text}\n"
            "Answer:"
        )
    elif task == "story_cloze":
        choices = choices or ["", ""]
        current_choices_text = f"(A) {choices[0]}\n(B) {choices[1]}"
        return (
            "Story: I wanted to make an omelet. I cracked two eggs into a bowl and whisked them. Then I poured them into a hot pan.\n"
            "(A) I ate a delicious omelet for breakfast.\n"
            "(B) I decided to order a pizza instead.\n"
            "End: I ate a delicious omelet for breakfast.\n\n"
            "Story: The runner tied his shoes tight. He sprinted as fast as he could during the race. He crossed the finish line first.\n"
            "(A) He was sad that he lost the race.\n"
            "(B) He won the gold medal.\n"
            "End: He won the gold medal.\n\n"
            f"Story: {question}\n"
            f"{current_choices_text}\n"
            "End:"
        )
    else:
        return question


# ---------------------------------------------------------------------------
# Sampling Strategies
# ---------------------------------------------------------------------------


def sample_with_strategies(
    logits: torch.Tensor,
    generated_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    is_3d = False
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    if logits.dim() == 3:
        is_3d = True
        batch_size, block_sz, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
    else:
        batch_size, vocab_size = logits.shape
        block_sz = 1

    target_gen_ids = None
    if generated_ids is not None and repetition_penalty != 1.0:
        target_gen_ids = generated_ids.repeat_interleave(block_sz, dim=0) if is_3d else generated_ids

    if target_gen_ids is not None:
        score = torch.gather(logits, 1, target_gen_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(1, target_gen_ids, score)

    if temperature < 1e-5:
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tokens.view(batch_size, block_sz) if is_3d else next_tokens.view(batch_size)

    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, top_k)
        min_values = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any():
        probs = torch.nan_to_num(probs, nan=1.0 / vocab_size)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens.view(batch_size, block_sz) if is_3d else next_tokens.view(batch_size)


# ---------------------------------------------------------------------------
# NA helpers (local to inference)
# ---------------------------------------------------------------------------


def _shape_tensor(lens: list[int], device: torch.device) -> torch.LongTensor:
    """Build a ``(B, 1)`` shape tensor from a Python list of per-sample lengths."""
    return torch.tensor([[int(l)] for l in lens], dtype=torch.long, device=device)


def _inference_autocast(device: torch.device):
    """Use bf16 autocast on the device selected for inference."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Main Inference Logic — NA form, no batch padding
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_task_repaint_inference(
    dit: ColaDiTModel,
    vae: ColaTextVAEModel,
    tokenizer: Tokenizer,
    prompts: list[dict],
    task_name: str = "lambada",
    device: torch.device = torch.device("cuda"),
    # Diffusion params
    T: float = 1000.0,
    timestep_num: int = 16,
    guidance_scale: float = 7.0,
    # Generation params
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    # Special token IDs (adjust to your tokenizer)
    pad_token_id: int = 100277,
    eos_token_id: Optional[int] = None,
    im_end_token_id: Optional[int] = None,
    is_sft: bool = False,
    im_start_token_id: Optional[int] = None,
    user_token_id: Optional[int] = None,
    assistant_token_id: Optional[int] = None,
    newline_token_id: Optional[int] = None,
) -> list[dict]:
    """End-to-end Cola DLM inference (Eq. 2.2.4–2.2.6 of the paper).

    Realizes the three-step inference algorithm of *Continuous Latent
    Diffusion Language Model* (arXiv:2605.06548) for a batch of
    prompts:

    1. **Prefix encode** ``z^pre ~ q_phi(z^pre | x^pre)`` via
       :meth:`ColaTextVAEModel.encode` (Eq. 2.2.4). The prefix latent is
       written into the per-layer KV cache of both the DiT prior and
       the VAE decoder so subsequent block-wise calls only pay
       attention over the newly added Q.
    2. **Block-wise latent prior transport** ``hat z_0^(b) =
       Phi^psi_{0<-1}(eps^(b); z^pre, hat z_0^(<b))`` (Eq. 2.2.5),
       implemented as a loop over generation blocks
       ``b = 1, 2, ..., B``. For each block we draw
       ``eps^(b) ~ N(0, I)``, then integrate the DiT vector field
       ``v_psi`` from ``t = T`` to ``t = 0`` along ``timesteps`` with an
       Euler-style update ``z_{t-Δ} = z_t - Δ/T * v_psi(...)`` and a
       conditional / unconditional CFG combination. The cleaned block
       is then committed to the prior + decoder KV cache.
    3. **Conditional decode** ``hat x^res ~ p_theta(x^res | z^pre,
       hat z_0^(1:B))`` (Eq. 2.2.6) via :meth:`ColaTextVAEModel.decode`,
       producing token logits one block at a time. Logits are sampled
       with the configured strategy (greedy / top-k / top-p /
       repetition penalty) and appended to each sample's output buffer
       until EOS or the ``max_new_tokens`` budget is hit.

    The whole pipeline (DiT prior + VAE encoder + VAE decoder) runs in
    NA / flatten-concat form: no ``max_len`` padding and therefore no
    ``rope_pad_offset`` or padding-block attention mask bookkeeping. The
    only practical detail layered on top of the paper's algorithm is a
    per-sample CFG fallback for prompts shorter than ``block_size``
    (whose first generation block has an empty prefix and would
    otherwise just amplify bf16 kernel noise); see the comment block in
    the loop body for details.
    """
    dit.eval()
    vae.eval()

    scale = vae.scaling_factor
    shift = vae.shifting_factor
    patch_size = vae.patch_size
    block_size = dit.block_size

    def _diffusion_dt(t_curr, t_next):
        return (float(t_curr) - float(t_next)) / max(T, 1.0)

    # -----------------------------------------------------------------
    # Step 1: tokenise + per-sample block-align pad (no batch pad)
    # -----------------------------------------------------------------
    batch_prompts_text: list[str] = []
    input_ids_list: list[torch.Tensor] = []
    token_labels_list: list[torch.Tensor] = []
    prompt_len_remainders: list[int] = []

    chunk = patch_size * block_size
    for item in prompts:
        prompt_str = apply_prompt_template(
            task=task_name,
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer=item.get("ground_truth", item.get("answer", "")),
            choices=item.get("choices", None),
        )
        batch_prompts_text.append(prompt_str)

        ids = tokenizer.encode(prompt_str).ids
        if is_sft and all(
            tid is not None
            for tid in [im_start_token_id, user_token_id, assistant_token_id, newline_token_id, im_end_token_id]
        ):
            ids = (
                [im_start_token_id, user_token_id, newline_token_id]
                + ids
                + [im_end_token_id, newline_token_id, im_start_token_id, assistant_token_id, newline_token_id]
            )

        if patch_size > 1:
            prompt_len_remainders.append(len(ids) % patch_size)

        p_pad_len = (chunk - len(ids) % chunk) % chunk
        t_labels = [1] * len(ids) + [3] * p_pad_len
        ids = ids + [pad_token_id] * p_pad_len

        input_ids_list.append(torch.tensor(ids, dtype=torch.long, device=device))
        token_labels_list.append(torch.tensor(t_labels, dtype=torch.long, device=device))

    batch_size = len(input_ids_list)

    # -----------------------------------------------------------------
    # Step 2: VAE encode (list-in, list-out)
    # -----------------------------------------------------------------
    # Posterior mode ``(mode() - shift) * scale`` in fp32 — matches the
    # trainer's ``latents_3d = ... .float()`` after the autocast block.
    with _inference_autocast(device):
        enc = vae.encode(input_ids_list)
        latents_list = [((lat - shift) * scale).float() for lat in enc.latents_list]

    # -----------------------------------------------------------------
    # Step 3: latent labels per sample + first-generation-block layout
    # -----------------------------------------------------------------
    # Derivation matches the trainer's ``reshape -> contains_1/contains_2``
    # priority rule: label 1 wins over 2 wins over 3.
    latent_labels_list: list[torch.Tensor] = []
    for t_labels in token_labels_list:
        n_patches = t_labels.shape[0] // patch_size
        reshaped = t_labels.view(n_patches, patch_size)
        c1 = (reshaped == 1).any(dim=1)
        c2 = (reshaped == 2).any(dim=1)
        lat = torch.full((n_patches,), 3, dtype=torch.long, device=device)
        lat[c2] = 2
        lat[c1] = 1
        latent_labels_list.append(lat)

    # Strip trailing label-3 pad: NA form carries only real (label 1/2)
    # latents from this point on.
    prompt_latent_counts: list[int] = [int((lat == 1).sum().item()) for lat in latent_labels_list]

    prefix_list: list[torch.Tensor] = []
    first_block_latents_list: list[torch.Tensor] = []
    first_block_labels_list: list[torch.Tensor] = []
    first_block_prompt_token_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    force_complete_prefix_only = False

    for i in range(batch_size):
        num_ones = prompt_latent_counts[i]
        lat_total_i = latents_list[i].shape[0]  # total latents *including* label-3 pad tail
        # Reference block taken from the original padded implementation:
        # the last ``block_size`` latents of the sample's full (padded)
        # latent sequence. Functions only as a placeholder — the main loop
        # always overwrites the first generation block with ``torch.randn``.
        pad_placeholder = latents_list[i][lat_total_i - block_size : lat_total_i].clone()

        if num_ones % block_size != 0:
            prefix_fill_latents = block_size - (num_ones % block_size)
            if max_new_tokens < prefix_fill_latents:
                force_complete_prefix_only = True
            start_idx = (num_ones // block_size) * block_size
            # ``start_idx + block_size`` should always fit, because the
            # per-sample pad guarantees at least one extra block, but
            # keep the same safety net as the trainer.
            if start_idx + block_size <= lat_total_i:
                block_latents = latents_list[i][start_idx : start_idx + block_size].clone()
                block_labels = latent_labels_list[i][start_idx : start_idx + block_size].clone()
                # Promote label 3 (was pad) → 2 (to-generate) for the first gen block.
                block_labels[block_labels == 3] = 2
                # Token count that the FIRST gen block copies from the prompt
                # — used to trim the same tokens off the prediction later.
                token_start = start_idx * patch_size
                token_end = min(token_start + block_size * patch_size, token_labels_list[i].shape[0])
                first_block_prompt_token_counts[i] = (token_labels_list[i][token_start:token_end] == 1).sum()
                prefix_list.append(latents_list[i][:start_idx].clone())
                first_block_latents_list.append(block_latents)
                first_block_labels_list.append(block_labels)
            else:
                prefix_list.append(latents_list[i][:num_ones].clone())
                first_block_latents_list.append(pad_placeholder)
                first_block_labels_list.append(torch.full((block_size,), 2, dtype=torch.long, device=device))
        else:
            prefix_list.append(latents_list[i][:num_ones].clone())
            first_block_latents_list.append(pad_placeholder)
            first_block_labels_list.append(torch.full((block_size,), 2, dtype=torch.long, device=device))

    # -----------------------------------------------------------------
    # Step 4: prefix KV prefetch (DiT + VAE decoder), NA form
    # -----------------------------------------------------------------
    timesteps = torch.linspace(int(T), 0, timestep_num + 1, dtype=torch.float32)

    # Enable KV cache on both models.
    for block in dit.blocks:
        block.set_kv_cache(True)
    vae.set_kv_cache(True)

    prefix_lens = [p.shape[0] for p in prefix_list]
    txt_shape_prefix = _shape_tensor(prefix_lens, device)

    # NA inference does not need a ``rope_pad_offset``: there is no batch
    # padding, so every sample is RoPE-indexed with canonical
    # ``k_offset=0`` and ``q_offset = txt_shape - txt_q_shape`` (Q aligned
    # to the tail of K). For the prefix prefetch call ``txt_q_shape ==
    # txt_shape`` so ``q_offset == 0``, i.e. positions ``0..L_prefix-1``
    # for both Q and K.
    #
    # We wrap every DiT forward in ``torch.autocast(..., bfloat16)`` to
    # match the native trainer's mixed-precision contract: master weights
    # stay in fp32 (so LayerNorm / RMSNorm gains are not silently
    # quantised) while matmul / conv inputs are downcast to bf16 per-op.
    if any(p.shape[0] > 0 for p in prefix_list):
        txt_prefix = torch.cat(prefix_list, dim=0).to(torch.bfloat16)
        ts_prefix = torch.zeros(txt_prefix.shape[0], device=device, dtype=torch.bfloat16)
        with _inference_autocast(device):
            _ = dit(
                txt=txt_prefix,
                txt_shape=txt_shape_prefix,
                txt_q_shape=txt_shape_prefix,
                timestep=ts_prefix,
                update_kv=True,
                use_kv_cache=True,
            )
        with _inference_autocast(device):
            _ = vae.decode(
                z=torch.cat(prefix_list, dim=0),
                txt_shape=txt_shape_prefix,
                txt_q_shape=txt_shape_prefix,
                update_kv=True,
            )

    # -----------------------------------------------------------------
    # Step 5: block-wise generation loop
    # -----------------------------------------------------------------
    # Per-sample cumulative K length (initially = prefix length).
    txt_shape_cum = _shape_tensor(prefix_lens, device)

    # Precomputed clean-guidance tensors: ``first_block_*_flatten[i*block_size:(i+1)*block_size]``
    # corresponds to the first generation block of sample ``i``.
    first_block_latents_flatten = torch.cat(first_block_latents_list, dim=0)  # (B*block_size, d)
    first_block_labels_flatten = torch.cat(first_block_labels_list, dim=0)  # (B*block_size,)
    flat_mask = first_block_labels_flatten == 1

    # Per-sample CFG scale for the first generation block. When a sample's
    # prompt is shorter than ``block_size`` tokens its prefix KV cache is
    # empty, so the conditional and unconditional DiT passes are
    # mathematically identical (same K = current block). Using the
    # configured ``guidance_scale`` would then just amplify bf16 kernel
    # noise by 7x and degrade the first few generated tokens. We detect
    # this per sample and fall back to ``scale = 1.0`` (no CFG) for the
    # first block only; from the second block onward every sample has a
    # non-empty prefix (the just-generated block) so the configured scale
    # is restored for the whole batch.
    # Kept in bf16 so the CFG combination stays in bf16 arithmetic (same
    # as the original ``guidance_scale * (cond - uncond) + uncond`` path
    # when ``guidance_scale`` was a Python scalar); this preserves
    # bitwise alignment with the trainer for samples whose prefix is
    # non-empty and only diverges for short-prompt samples by design.
    cfg_scale_first_block = (
        torch.tensor(
            [guidance_scale if pl > 0 else 1.0 for pl in prefix_lens],
            device=device,
            dtype=torch.bfloat16,
        )
        .repeat_interleave(block_size)
        .unsqueeze(-1)
    )  # (B*block_size, 1)
    _empty_prefix_samples = [i for i, pl in enumerate(prefix_lens) if pl == 0]
    if _empty_prefix_samples:
        print(
            f"[inference] {len(_empty_prefix_samples)}/{batch_size} sample(s) have "
            f"prompts shorter than block_size={block_size}; CFG is disabled for "
            f"their first block (guidance_scale -> 1.0) to avoid amplifying "
            f"bf16 noise. Subsequent blocks use guidance_scale={guidance_scale}."
        )

    context_ids: Optional[torch.Tensor] = None
    step = 0
    stop_flag = False
    eos_status = torch.zeros(batch_size, dtype=torch.bool, device=device)

    txt_q_shape = _shape_tensor([block_size] * batch_size, device)

    while not stop_flag:
        # Each iteration appends one ``block_size``-sized block to every
        # sample's K length.
        txt_shape_cum = txt_shape_cum + block_size

        latent_dim = first_block_latents_flatten.shape[-1]

        # --- Initial noise for this block ----------------------------------
        # By default the noise is drawn from the global RNG (so generation
        # is stochastic unless the caller seeds ``torch.manual_seed``
        # outside this function). Setting ``COLA_INFER_PER_SAMPLE_NOISE_SEED``
        # to an integer switches to deterministic per-sample seeding keyed
        # by ``(base_seed, sample_id, step)``: sample X always gets the
        # same noise regardless of which row of the batch tensor it lands
        # on, which is useful for reproducible accuracy comparisons across
        # different batch sizes / world sizes.
        per_sample_seed_env = os.environ.get("COLA_INFER_PER_SAMPLE_NOISE_SEED", "").strip()
        if per_sample_seed_env:
            base_seed = int(per_sample_seed_env)
            noise_3d = torch.empty(batch_size, block_size, latent_dim, device=device)
            for b, item in enumerate(prompts):
                sid = item.get("id")
                if sid is None:
                    sid = b
                try:
                    sid_int = int(sid)
                except (TypeError, ValueError):
                    import zlib

                    sid_int = zlib.crc32(str(sid).encode("utf-8")) & 0xFFFFFFFF
                g = torch.Generator(device=device)
                g.manual_seed(base_seed + sid_int * 1_000 + int(step) * 10_000_000)
                noise_3d[b] = torch.randn(block_size, latent_dim, device=device, generator=g)
            txt = noise_3d.view(batch_size * block_size, latent_dim)
        else:
            txt = torch.randn(batch_size * block_size, latent_dim, device=device)

        for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
            ts_batch = torch.full((txt.shape[0],), t_curr, device=device)
            dt = _diffusion_dt(t_curr, t_next)

            if step == 0:
                # Clean-guidance: keep real prompt positions at t=0 and
                # pin their latent to the ground-truth value throughout.
                ts_batch[flat_mask] = 0
                txt[flat_mask] = first_block_latents_flatten[flat_mask]

            txt_bf16 = txt.to(torch.bfloat16)
            ts_bf16 = ts_batch.to(torch.bfloat16)

            with _inference_autocast(device):
                drift_cond = dit(
                    txt=txt_bf16,
                    txt_shape=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    timestep=ts_bf16,
                    update_kv=False,
                    use_kv_cache=True,
                ).txt_sample

                # Unconditional branch: each sample attends only to its own
                # current block (K = Q = block_size per sample, no cache).
                drift_uncond = dit(
                    txt=txt_bf16,
                    txt_shape=txt_q_shape,
                    txt_q_shape=txt_q_shape,
                    timestep=ts_bf16,
                    update_kv=False,
                    use_kv_cache=False,
                ).txt_sample

            # Keep CFG combination in bf16 to match the FSDP trainer
            # numerics; ``txt - drift * dt`` then auto-promotes to fp32
            # because ``txt`` is fp32. For the first block we use the
            # per-sample scale tensor built above (empty-prefix samples
            # get scale=1 so CFG is effectively skipped for them).
            s = cfg_scale_first_block if step == 0 else guidance_scale
            drift = s * (drift_cond - drift_uncond) + drift_uncond
            txt_next = txt - drift * dt

            if step == 0:
                txt_next[flat_mask] = first_block_latents_flatten[flat_mask]

            txt = txt_next

        # ------------ decode one block via VAE (NA) ----------------
        with _inference_autocast(device):
            decoded = vae.decode(
                z=txt,
                txt_shape=txt_shape_cum,
                txt_q_shape=txt_q_shape,
                update_kv=True,
            )
        # ``decoded`` is (1, B*block_size*patch_size, vocab) — same
        # per-sample ordering as the NA layout, reshape to (B, block_size*patch_size, vocab).
        decoded_logits = decoded.view(batch_size, block_size * patch_size, -1)

        one_block_ids = sample_with_strategies(
            decoded_logits,
            generated_ids=context_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if context_ids is None:
            context_ids = one_block_ids
        else:
            context_ids = torch.cat([context_ids, one_block_ids], dim=1)

        for b in range(one_block_ids.shape[0]):
            if eos_token_id is not None and eos_token_id in one_block_ids[b]:
                eos_status[b] = True
            if im_end_token_id is not None and im_end_token_id in one_block_ids[b]:
                eos_status[b] = True
        if eos_status.all():
            stop_flag = True

        # -------- write the just-denoised block into the DiT KV cache --------
        txt_bf16 = txt.to(torch.bfloat16)
        with _inference_autocast(device):
            _ = dit(
                txt=txt_bf16,
                txt_shape=txt_shape_cum,
                txt_q_shape=txt_q_shape,
                timestep=torch.zeros(txt.shape[0], device=device, dtype=torch.bfloat16),
                update_kv=True,
                use_kv_cache=True,
            )

        step += 1
        if force_complete_prefix_only and step >= 1:
            stop_flag = True
        elif step * block_size * patch_size >= max_new_tokens:
            stop_flag = True

    # Clean up KV cache
    for block in dit.blocks:
        block.set_kv_cache(False)
    vae.set_kv_cache(False)

    # -----------------------------------------------------------------
    # Step 6: trim the leading prompt slice out of each sample's output
    # -----------------------------------------------------------------
    context_ids_cpu = context_ids.detach().cpu()
    prompt_trim_counts = first_block_prompt_token_counts.detach().cpu().tolist()
    trimmed_ids = []
    for sample_idx, trim_count in enumerate(prompt_trim_counts):
        trim_count = max(0, min(int(trim_count), context_ids_cpu.shape[1]))
        trimmed_ids.append(context_ids_cpu[sample_idx, trim_count:].tolist())
    generated_texts = tokenizer.decode_batch(trimmed_ids, skip_special_tokens=False)

    results: list[dict] = []
    for idx, gen_text in enumerate(generated_texts):
        entry = {
            "id": prompts[idx].get("id"),
            "prompt": batch_prompts_text[idx],
            "generate": gen_text,
            "ground_truth": prompts[idx].get("answer", prompts[idx].get("ground_truth", "")),
        }
        if prompts[idx].get("choices"):
            entry["choices"] = prompts[idx]["choices"]
        if patch_size > 1 and idx < len(prompt_len_remainders):
            entry["prompt_len_mod_patch_size"] = prompt_len_remainders[idx]
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cola Text DiT + VAE Inference (NA form)")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to DiT model directory")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE model directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--task_name", type=str, default="lambada", help="Task name for prompt template")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to process (0 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens per sample")
    parser.add_argument("--timestep_num", type=int, default=16, help="Number of diffusion timesteps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="CFG guidance scale")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--pad_token_id", type=int, default=100277, help="Pad token ID")
    parser.add_argument(
        "--eos_token_id", type=int, default=None, help="EOS token ID (default: None → stop only on max_new_tokens)"
    )
    parser.add_argument(
        "--im_end_token_id",
        type=int,
        default=None,
        help="im_end token ID (default: None → stop only on max_new_tokens)",
    )
    parser.add_argument("--rank", type=int, default=0, help="GPU rank for multi-GPU data-parallel inference")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs for data-parallel inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading DiT from {args.dit_path}...")
    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device)

    print(f"Loading VAE from {args.vae_path}...")
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device)

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    raw_data = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    if args.max_samples > 0:
        raw_data = raw_data[: args.max_samples]

    total = len(raw_data)

    # Simple stride-based data-parallel sharding across GPUs.
    my_data = raw_data[args.rank :: args.world_size]
    print(f"[Rank {args.rank}/{args.world_size}] " f"{len(my_data)}/{total} samples for task '{args.task_name}'")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.world_size > 1:
        output_file = os.path.join(args.output_dir, f"{args.task_name}_rank{args.rank}.jsonl")
    else:
        output_file = os.path.join(args.output_dir, f"{args.task_name}.jsonl")

    if not my_data:
        print(f"[Rank {args.rank}] No samples to process, skipping.")
        with open(output_file, "w") as f:
            pass
        return

    all_results = []
    for batch_start in range(0, len(my_data), args.batch_size):
        batch_data = my_data[batch_start : batch_start + args.batch_size]
        print(f"  [Rank {args.rank}] " f"batch {batch_start // args.batch_size + 1} " f"({len(batch_data)} samples)")
        batch_results = generate_task_repaint_inference(
            dit=dit,
            vae=vae,
            tokenizer=tokenizer,
            prompts=batch_data,
            task_name=args.task_name,
            device=device,
            T=1000.0,
            timestep_num=args.timestep_num,
            guidance_scale=args.guidance_scale,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=args.pad_token_id,
            eos_token_id=args.eos_token_id,
            im_end_token_id=args.im_end_token_id,
        )
        all_results.extend(batch_results)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done! {len(all_results)} results saved to {output_file}")


if __name__ == "__main__":
    main()
