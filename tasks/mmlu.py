# Adapted from nanochat.

from datasets import load_dataset
from .common import Task, render_mc


class MMLU(Task):
    letters = ("A", "B", "C", "D")

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"]
        assert split in ["auxiliary_train", "validation", "dev", "test"]
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        user_message = render_mc(row["question"], self.letters, row["choices"])
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": self.letters[row["answer"]]},
            ]
        }
