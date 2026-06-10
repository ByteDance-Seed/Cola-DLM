# Adapted from nanochat.
# Loads conversations from a JSONL file.

import os
import json
import urllib.request
from .common import Task


class CustomJSON(Task):
    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    messages = json.loads(line)
                    assert isinstance(messages, list) and len(messages) >= 2
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}
