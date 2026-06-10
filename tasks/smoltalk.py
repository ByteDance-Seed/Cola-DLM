# Adapted from nanochat.

from datasets import load_dataset
from .common import Task


class SmolTalk(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        return {"messages": row["messages"]}
