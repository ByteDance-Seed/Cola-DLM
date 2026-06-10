# Adapted from nanochat.

import os
import random
import urllib.request

from .common import Task

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# User message templates
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "In the word {word}, how many {letter} are there",
    "Count how many {letter} appear in {word}",
    "How many {letter}s are in {word}",
    "Count the {letter} in {word}",
    "How many {letter} does {word} have",
]


def _download_word_list():
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/nanochat"))
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "words_alpha.txt")
    if not os.path.exists(path):
        print(f"Downloading {WORD_LIST_URL}...")
        urllib.request.urlretrieve(WORD_LIST_URL, path)
        print(f"Downloaded to {path}")
    return path


class SpellingBee(Task):
    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.size = size
        with open(_download_word_list()) as f:
            self.words = [line.strip() for line in f]

    def num_examples(self):
        return self.size

    def get_example(self, index):
        rng = random.Random(index)
        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        user_msg = template.format(letter=letter, word=word)
        if rng.random() < 0.5:
            user_msg += "?"

        word_letters = ",".join(list(word))
        manual_text = (
            f"We are asked to find the number '{letter}' in the word '{word}'. "
            f"Let me try a manual approach first.\n\n"
            f"First spell the word out:\n{word}:{word_letters}\n\n"
            f"Then count the occurrences of '{letter}':\n"
        )
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"
        manual_text += f"\nThis gives us {running_count}."

        assistant_parts = [
            {"type": "text", "text": manual_text},
            {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"},
            {"type": "python", "text": f"'{word}'.count('{letter}')"},
            {"type": "python_output", "text": str(count)},
            {"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ]
        }


class SimpleSpelling(Task):
    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.size = size
        with open(_download_word_list()) as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    def num_examples(self):
        return self.size

    def get_example(self, index):
        rng = random.Random(index)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        return {
            "messages": [
                {"role": "user", "content": f"Spell the word: {word}"},
                {"role": "assistant", "content": f"{word}:{word_letters}"},
            ]
        }
