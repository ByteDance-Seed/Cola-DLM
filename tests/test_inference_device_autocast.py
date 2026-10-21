from __future__ import annotations

from types import SimpleNamespace

import torch

from cola_dlm.inference import generate_task_repaint_inference


class _FakeCacheBlock:
    def set_kv_cache(self, enabled: bool) -> None:
        pass


class _FakeDiT:
    def __init__(self) -> None:
        self.block_size = 1
        self.blocks = [_FakeCacheBlock()]
        self.projection = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)

    def eval(self) -> _FakeDiT:
        return self

    def __call__(self, txt: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(txt_sample=self.projection(txt))


class _FakeVAE:
    def __init__(self) -> None:
        self.scaling_factor = 1.0
        self.shifting_factor = 0.0
        self.patch_size = 1
        self.decoder = SimpleNamespace(in_layer=torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16))

    def eval(self) -> _FakeVAE:
        return self

    def encode(self, input_ids_list: list[torch.Tensor]) -> SimpleNamespace:
        latents = [
            torch.ones((input_ids.shape[0], 2), dtype=torch.bfloat16, device=input_ids.device)
            for input_ids in input_ids_list
        ]
        return SimpleNamespace(latents_list=latents)

    def decode(
        self,
        z: torch.Tensor,
        txt_shape: torch.Tensor,
        txt_q_shape: torch.Tensor,
        update_kv: bool = False,
    ) -> torch.Tensor:
        hidden = self.decoder.in_layer(z)
        return torch.cat((hidden, -hidden), dim=-1).unsqueeze(0)

    def set_kv_cache(self, enabled: bool) -> None:
        pass


class _FakeTokenizer:
    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[1])

    def decode_batch(self, sequences: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        return ["answer"] * len(sequences)


def test_cpu_inference_autocasts_bfloat16_weights() -> None:
    results = generate_task_repaint_inference(
        dit=_FakeDiT(),
        vae=_FakeVAE(),
        tokenizer=_FakeTokenizer(),
        prompts=[{"question": "What is the answer?"}],
        task_name="lambada",
        device=torch.device("cpu"),
        T=1.0,
        timestep_num=1,
        guidance_scale=1.0,
        max_new_tokens=1,
        temperature=0.0,
        pad_token_id=0,
    )

    assert results[0]["generate"] == "answer"
