"""
Distributed dataloader for pretraining on parquet text datasets.

BOS-aligned best-fit packing (adapted from nanochat):
- Every row starts with a BOS/document-separator token
- Documents packed using best-fit to minimize cropping
- When no document fits, crops a document to fill exactly
- 100% utilization (no padding), ~35% tokens cropped at T=2048
"""

import os
import torch
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def list_parquet_files(data_dir):
    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(data_dir, f) for f in parquet_files]


def _get_dist_info():
    if "RANK" in os.environ:
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1


def _document_batches(data_dir, split, resume_state_dict, batch_size):
    """Infinite iterator over document text batches from parquet files."""
    _, ddp_rank, _, ddp_world_size = _get_dist_info()

    parquet_paths = list_parquet_files(data_dir)
    assert parquet_paths, f"No parquet files found in {data_dir}"
    # last shard = val, rest = train
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            pf = pq.ParquetFile(parquet_paths[pq_idx])
            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                base_idx = resume_rg_idx // ddp_world_size + 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                for i in range(0, len(texts), batch_size):
                    yield texts[i : i + batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def pretrain_data_loader(
    tokenizer_path,
    data_dir,
    B,
    T,
    split="train",
    bos_token_id=100257,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    buffer_size=1000,
):
    """
    BOS-aligned best-fit packing dataloader for pretraining.

    Args:
        tokenizer_path: path to tokenizer.json (OLMo 2 / HuggingFace tokenizers format)
        data_dir: directory containing parquet shards
        B: batch size (per device)
        T: sequence length (each row has T+1 tokens: T inputs + 1 for shifted target)
        split: "train" or "val"
        bos_token_id: document separator token id (<|endoftext|> = 100257)
        device: target device

    Yields:
        (inputs, targets, state_dict) where inputs/targets are (B, T) on device
    """
    assert split in ("train", "val")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    row_capacity = T + 1
    batches = _document_batches(data_dir, split, resume_state_dict, tokenizer_batch_size)
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        encoded = tokenizer.encode_batch(doc_batch)
        for enc in encoded:
            doc_buffer.append([bos_token_id] + enc.ids)

    use_cuda = device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda")
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # best-fit: pick largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    dl = len(doc)
                    if dl <= remaining and dl > best_len:
                        best_idx = i
                        best_len = dl

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # crop shortest doc to fill remaining space
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict
