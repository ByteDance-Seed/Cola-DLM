<div align="center">

# Cola DLM

**Continuous Latent Diffusion Language Model — a hierarchical latent-space text diffusion model with a block-causal DiT prior over a Text VAE.**

[![arXiv](https://img.shields.io/badge/arXiv-2605.06548-b31b1b.svg)](https://arxiv.org/abs/2605.06548)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/ByteDance-Seed/Cola-DLM)
[![HuggingFace Daily](https://img.shields.io/badge/HF-Daily%20Paper-yellow.svg)](https://huggingface.co/papers/2605.06548)
[![Project Page](https://img.shields.io/badge/Project%20Page-Cola--DLM-1e90ff.svg)](https://hongcanguo.github.io/Cola-DLM/)
[![Blog](https://img.shields.io/badge/Blog-Post-1e90ff.svg)](https://hongcanguo.github.io/posts/2026-cola-dlm.html)
[![Zhihu](https://img.shields.io/badge/Zhihu-Article-0084ff.svg)](https://zhuanlan.zhihu.com/p/2038324180920313704)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.40%2B-yellow.svg)](https://github.com/huggingface/transformers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](README.md) · [中文](README_zh.md)

</div>

> **Cola DLM** (`Co`ntinuous `La`tent `D`iffusion `L`anguage `M`odel) is the official, HuggingFace-Transformers-compatible open-source release of the paper [*Continuous Latent Diffusion Language Model*](https://arxiv.org/abs/2605.06548). Cola DLM is a **hierarchical latent-variable language model**: a *Text VAE* learns a stable mapping `q_phi(z_0 | x)` between text and a continuous latent sequence; a *block-causal Diffusion Transformer (DiT)* models the latent prior `p_psi(z_0)` via Flow Matching; and the *conditional decoder* `p_theta(x | z_0)` realizes the actual tokens. From a unified Markov-path perspective, the diffusion process performs **latent prior transport** rather than token-level observation recovery, separating global semantic organization from local textual realization. This repository ships the trained checkpoint together with a no-padding ("NA") flatten-concat inference pipeline that runs natively under HuggingFace Transformers.

---

## Paper

- **Title:** Continuous Latent Diffusion Language Model
- **Authors:** Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng (ByteDance Seed et al.)
- **arXiv:** [arxiv.org/abs/2605.06548](https://arxiv.org/abs/2605.06548)
- **Model weights:** [huggingface.co/ByteDance-Seed/Cola-DLM](https://huggingface.co/ByteDance-Seed/Cola-DLM)
- **HuggingFace daily paper:** [huggingface.co/papers/2605.06548](https://huggingface.co/papers/2605.06548)
- **Project page:** [hongcanguo.github.io/Cola-DLM](https://hongcanguo.github.io/Cola-DLM/)
- **Blog post:** [hongcanguo.github.io/posts/2026-cola-dlm.html](https://hongcanguo.github.io/posts/2026-cola-dlm.html)
- **Zhihu article:** [zhuanlan.zhihu.com/p/2038324180920313704](https://zhuanlan.zhihu.com/p/2038324180920313704)

---

## Method at a glance

<p align="center">
  <img src="docs/figures/cola_main_fig.png" alt="Overall workflow of Cola DLM: Text VAE pretraining, joint Text VAE + block-causal Text DiT training, and KV-cached inference." width="900"/>
</p>

<p align="center"><em><strong>Figure 1 — Overall workflow of Cola DLM.</strong> <strong>Stage 1</strong>: Text VAE pretraining with reconstruction, BERT-style masking, and a KL regularizer to a base prior. <strong>Stage 2</strong>: joint Text VAE + block-causal Text DiT training; the DiT learns the latent prior <code>p_psi(z_0)</code> via Flow Matching under the visible set <code>V_b</code>. <strong>Inference</strong>: prefix encoding <code>q_phi(z<sup>pre</sup> | x<sup>pre</sup>)</code>, block-wise prior transport <code>Phi<sup>psi</sup><sub>0←1</sub></code> in latent space, and conditional decoding <code>p_theta(x | z_0)</code> with KV cache.</em></p>

Cola DLM defines the joint generative distribution as

```
p(x, z_0) = p_theta(x | z_0) * p_psi(z_0),    p(x) = ∫ p_theta(x | z_0) * p_psi(z_0) dz_0,
```

where `q_phi(z_0 | x)` is an inference (encoder) model used only at training and prefix-encoding time. The latent is decomposed into `B` blocks `z_0 = (z_0^(1), ..., z_0^(B))` with a block-causal factorization `p_psi(z_0) = p_psi(z_0^(1)) * ∏_{b≥2} p_psi(z_0^(b) | z_0^(<b))`, which directly mirrors the block-causal attention pattern of the DiT.

Training is two-stage:

1. **Stage 1 — Text VAE pretraining.** Learns a stable text↔latent mapping (`q_phi`, `p_theta`) with reconstruction, BERT-style masking, and a KL regularizer to a base prior `p_base`.
2. **Stage 2 — Joint Text VAE + block-causal Text DiT pretraining.** The DiT learns the latent prior `p_psi(z_0)` via conditional **Flow Matching** under the visible set `V_b = {sg(z_0^(<b)), z_t^(b)}`, while the VAE remains trainable under recon, mask, and a reference-encoder KL regularizer that prevents latent drift.

Inference (this repo) implements the paper's three-step recipe: **(i) prefix encode** `z^pre ~ q_phi(z^pre | x^pre)`; **(ii) block-wise generation** by transporting noise under the historical condition, `hat z_0^(b) = Phi^psi_{0←1}(eps^(b); z^pre, hat z_0^(<b))`, with `eps^(b) ~ N(0, I)`; **(iii) conditional decoding** `hat x^res ~ p_theta(x^res | z^pre, hat z_0^(1:B))`. See [`docs/architecture.md`](docs/architecture.md) for the full mapping between paper notation and code paths.

---

## Table of contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [OpenAI-compatible deployment](#openai-compatible-deployment)
- [Evaluation benchmarks](#evaluation-benchmarks)
- [Unified text–image (preliminary)](#unified-textimage-preliminary)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **Hierarchical latent-variable model.** `ColaTextVAEModel` provides the inference encoder `q_phi` and the conditional decoder `p_theta`; `ColaDiTModel` parameterizes the block-causal latent prior `p_psi`. Diffusion is used to *transport the latent prior* (Eq. 2.1.4 of the paper), not to recover tokens.
- **HuggingFace-native.** `ColaDiTModel` and `ColaTextVAEModel` subclass `transformers.PreTrainedModel` and ship with matching `PretrainedConfig` classes, so `from_pretrained` / `save_pretrained` / `AutoConfig` all work out of the box.
- **No-padding ("NA") inference.** Variable-length samples are concatenated along a single sequence axis with a companion `txt_shape: (B, 1)` describing per-sample lengths. RoPE, attention masks and the prior-transport loop are all driven by those lengths — no `max_len` padding is allocated at any point.
- **Block-causal prior + classifier-free guidance.** The DiT realizes one block of `Phi^psi_{0←1}` per generation step under the block-causal visibility constraint `V_b`, alternating a conditional (prefix-aware) and unconditional (empty-prefix) pass exactly like the training objective.
- **KV cache everywhere.** Both the DiT and the VAE decoder cache per-sample K/V projections between blocks, so generating block `t+1` only pays attention to the newly appended block's Q.
- **OpenAI-compatible serving.** [`openai_adapter/`](openai_adapter/) exposes Cola DLM through `POST /v1/chat/completions`, making it easy to deploy behind existing OpenAI-style clients, gateways, and evaluation tools.
- **Reproducible benchmark.** [`scripts/run_benchmark.sh`](scripts/run_benchmark.sh) reproduces the 8-task evaluation (LAMBADA, MMLU, OBQA, HellaSwag, RACE, SIQA, SQuAD, Story Cloze) reported in the paper's RQ4 with a single shell command, including multi-GPU data-parallel sharding.

See [`docs/architecture.md`](docs/architecture.md) and [`docs/model_card.md`](docs/model_card.md) for a deeper technical description.

---

## Installation

Cola DLM targets **Python 3.9+** and **PyTorch 2.1+** on Linux / macOS.

### From source (recommended)

```bash
git clone https://github.com/your-org/cola-dlm.git
cd cola-dlm

# Editable install with runtime dependencies
pip install -e .

# Or with dev extras (pytest, ruff, black, pre-commit)
pip install -e ".[dev]"
```

### From PyPI (once published)

```bash
pip install cola-dlm
```

---

## Quickstart

### 1. Prepare model weights

Download the HuggingFace-format model weights from [ByteDance-Seed/Cola-DLM](https://huggingface.co/ByteDance-Seed/Cola-DLM), or place compatible local weights under `hf_models/cola_dlm/`:

```
hf_models/
├── cola_dlm/
│   ├── cola_dit/        # config.json + model.safetensors
│   └── cola_vae/        # config.json + model.safetensors
└── tokenizer.json
```

### 2. Programmatic inference

```python
import torch
from tokenizers import Tokenizer
from cola_dlm import (
    ColaDiTModel,
    ColaTextVAEModel,
    generate_task_repaint_inference,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dit = ColaDiTModel.from_pretrained("hf_models/cola_dlm/cola_dit").to(device)
vae = ColaTextVAEModel.from_pretrained("hf_models/cola_dlm/cola_vae").to(device)
tokenizer = Tokenizer.from_file("hf_models/tokenizer.json")

prompts = [{"question": "Question: What is the capital of France? Answer:"}]
results = generate_task_repaint_inference(
    dit=dit,
    vae=vae,
    tokenizer=tokenizer,
    prompts=prompts,
    task_name="lambada",
    device=device,
    max_new_tokens=32,
    temperature=0.0,
    guidance_scale=7.0,
    timestep_num=16,
    pad_token_id=100277,
)
print(results[0]["generate"])
```

`generate_task_repaint_inference` implements the paper's inference algorithm end-to-end: (i) prefix encode through the Text VAE, (ii) block-wise prior transport through the block-causal DiT, (iii) conditional decoding back to tokens. See [`examples/quickstart.py`](examples/quickstart.py) for a runnable, end-to-end script.

### 3. CLI inference

```bash
cola-dlm-infer \
    --dit_path hf_models/cola_dlm/cola_dit \
    --vae_path hf_models/cola_dlm/cola_vae \
    --tokenizer_path hf_models/tokenizer.json \
    --input_jsonl generate_task_data/lambada.jsonl \
    --output_dir eval_output/my_run \
    --task_name lambada
```

Run `cola-dlm-infer --help` for the full argument list.

---

## OpenAI-compatible deployment

The [`openai_adapter/`](openai_adapter/) directory adds a lightweight HTTP service for serving Cola DLM through an OpenAI-compatible Chat Completions API:

```text
POST /v1/chat/completions
```

Install the adapter dependencies from the repository root:

```bash
pip install -e .
pip install -r openai_adapter/requirements.txt
```

Then start the service with the model paths and optional bearer token:

```bash
export COLA_DIT_PATH=hf_models/cola_dlm/cola_dit
export COLA_VAE_PATH=hf_models/cola_dlm/cola_vae
export COLA_TOKENIZER_PATH=hf_models/tokenizer.json
export COLA_MODEL_NAME=cola-dlm
export COLA_API_KEY=change-me

uvicorn openai_adapter.server:app --host 0.0.0.0 --port 8000
```

The service supports `GET /health`, `GET /v1/models`, and non-streaming `POST /v1/chat/completions`. See [`openai_adapter/README.md`](openai_adapter/README.md) for request examples, environment variables, and production notes.

---

## Evaluation benchmarks

<p align="center">
  <img src="docs/figures/rq4_scaling_vs_ar_llada.png" alt="Scaling curves across 8 benchmarks plus Task Average — Cola DLM (red) vs AR (blue) and LLaDA (orange), up to ~2000 EFLOPs." width="900"/>
</p>

<p align="center"><em><strong>Figure 2 — RQ4 headline scaling result.</strong> Strictly matched ~2B-parameter setup, unified <em>generative</em> evaluation protocol, scaling curves up to ~2000 EFLOPs across 8 benchmarks plus Task Average. <strong>Cola DLM (red)</strong> reaches the best final Task Average — and the curve is <em>still rising</em> — with a clear lead on reasoning-heavy <strong>MMLU, RACE, Story Cloze, OBQA</strong>; SQuAD eventually surpasses AR and approaches LLaDA's strong region. The result is conservative: latent dimension <code>d=16</code>, no extended training, room to scale further.</em></p>

The `scripts/` folder contains a one-click reproduction of the 8-task evaluation pipeline used in the paper's RQ4 scaling comparison:

```bash
# Evaluate all 8 tasks (assumes hf_models/ and generate_task_data/ are populated)
bash scripts/run_benchmark.sh

# Single task, single GPU
TASKS="lambada" NUM_GPUS=1 bash scripts/run_benchmark.sh

# Compute accuracy from evaluation outputs
python scripts/acc_calc.py
```

Reference accuracy numbers (see [`eval_output/accuracy_summary.csv`](eval_output/accuracy_summary.csv)):

| Task       | Accuracy (%) |
|------------|--------------|
| LAMBADA    | 50.80        |
| MMLU       | 19.30        |
| OBQA       | 23.00        |
| HellaSwag  | 10.70        |
| RACE       | 19.60        |
| SIQA       | 28.90        |
| SQuAD      | 30.90        |
| Story Cloze| 30.77        |
| **Tasks Average** | **26.75** |

> **Note on open-source model and accuracy:**
> The released model weights correspond to the **2000 EFLOPs** entry on the scaling curve in the paper's RQ4 — the largest training-compute checkpoint reported. Because the internal architecture used for evaluation in the paper differs slightly from the open-source HuggingFace Transformers-based implementation in this repository, per-task accuracy numbers may exhibit minor fluctuations, but the overall trend is consistent with the paper. Notably, the **Tasks Average (26.75%) measured here is slightly higher than the final average reported in the paper**.

---

## Unified text–image (preliminary)

<p align="center">
  <img src="docs/figures/unified_samples.png" alt="Unified text-image qualitative samples: text-only continuation, image-conditioned text generation, text-to-image samples, and shared MMDiT prior schematic." width="900"/>
</p>

<p align="center"><em><strong>Figure 3 — Towards unified text–image modeling.</strong> Modality-specific VAE encoders/decoders interface with a <em>shared</em> block-causal MMDiT prior over a joint latent state — the same hierarchical latent decomposition extends naturally from text to vision. <em>Left</em>: text-only continuation and image-conditioned text generation (image-to-text). <em>Middle</em>: text-to-image samples from in-house pretraining only (no SFT, no high-quality data curation). <em>Right</em>: schematic of the shared block-causal MMDiT prior. This is intentionally early-stage; comprehensive unified multimodal training is left for future work — see the paper's Discussion for the full set of qualitative samples.</em></p>

> The released open-source code in this repository covers the **text-only** Cola DLM pipeline (Text VAE + block-causal DiT prior). Unified text–image training and inference are reported in the paper's Discussion as preliminary experiments and are not included in this release.

---

## Project layout

```
cola-dlm/
├── cola_dlm/                 # Importable Python package
│   ├── __init__.py           # Public API re-exports
│   ├── configuration_cola_dit.py   # ColaDiTConfig — block-causal DiT prior knobs
│   ├── configuration_cola_vae.py   # ColaTextVAEConfig — Text VAE knobs
│   ├── modeling_cola_dit.py  # ColaDiTModel — block-causal DiT prior p_psi(z_0)
│   ├── modeling_cola_vae.py  # ColaTextVAEModel — encoder q_phi + decoder p_theta
│   ├── attention_utils.py    # NA flatten-concat helpers + block-causal mask (visible set V_b)
│   └── inference.py          # Batch benchmark CLI + generate_task_repaint_inference
├── docs/                     # Architecture, model card, inference docs
├── examples/                 # Minimal runnable examples
├── openai_adapter/            # OpenAI-compatible HTTP serving adapter
├── scripts/                  # Shell entry points (benchmark + accuracy)
├── tests/                    # Unit + smoke tests
├── eval_output/              # Reference benchmark outputs (CSV summary committed)
├── generate_task_data/       # Benchmark JSONL datasets
├── pyproject.toml            # Build + metadata + dep spec
├── requirements.txt          # Pinned runtime deps
├── LICENSE                   # Apache-2.0
├── NOTICE                    # Apache-2.0 attribution
├── SECURITY.md               # Vulnerability reporting
└── README.md / README_zh.md  # Project documentation
```

---

## Documentation

Long-form docs live under [`docs/`](docs/):

- [`docs/architecture.md`](docs/architecture.md) — hierarchical latent-variable framing, VAE + DiT architecture, block-wise prior-transport loop, CFG, NA flatten-concat layout, Stage 1 / Stage 2 training reference.
- [`docs/model_card.md`](docs/model_card.md) — intended use, training data, limitations, bias, responsible-AI notes.
- [`docs/inference.md`](docs/inference.md) — how to run batch benchmarks and the Python API.
- [`openai_adapter/README.md`](openai_adapter/README.md) — how to deploy the OpenAI-compatible HTTP service.

For security-sensitive reports, please follow [`SECURITY.md`](SECURITY.md).

---

## Star History

<a href="https://www.star-history.com/?repos=ByteDance-Seed%2FCola-DLM&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ByteDance-Seed/Cola-DLM&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ByteDance-Seed/Cola-DLM&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ByteDance-Seed/Cola-DLM&type=date&legend=top-left" />
 </picture>
</a>

---

## Citation

If Cola DLM contributes to your research, please cite the paper:

```bibtex
@article{guo2026cola,
  title   = {Continuous Latent Diffusion Language Model},
  author  = {Guo, Hongcan and Zhao, Qinyu and Zhao, Yian and Nie, Shen and
             Zhu, Rui and Guo, Qiushan and Wang, Feng and Yang, Tao and
             Zhao, Hengshuang and Wei, Guoqiang and Zeng, Yan},
  journal = {arXiv preprint arXiv:2605.06548},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.06548},
}
```

You may additionally cite this open-source release:

```bibtex
@software{cola_dlm_2026,
  title   = {Cola DLM: Official Open-Source Inference Code for Continuous Latent Diffusion Language Model},
  year    = {2026},
  url     = {[https://github.com/your-org/cola-dlm](https://github.com/ByteDance-Seed/Cola-DLM)},
  version = {0.1.0}
}
```

---

## License

Cola DLM is released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions.
