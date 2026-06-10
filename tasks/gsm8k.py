# Adapted from nanochat.

import re
from datasets import load_dataset
from .common import Task


class GSM8K(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"]
        assert split in ["train", "test"]
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        assistant_message_parts = []
        parts = re.split(r"(<<[^>]+>>)", answer)
        for part in parts:
            if part.startswith("<<") and part.endswith(">>"):
                inner = part[2:-2]
                if "=" in inner:
                    expr, result = inner.rsplit("=", 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_message_parts},
            ]
        }
