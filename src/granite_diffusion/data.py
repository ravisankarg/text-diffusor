import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from .formatting import DEFAULT_TEMPLATE, DiffusionTemplate


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    answer_start: int


def encode_example(
    tokenizer,
    instruction: str,
    response: str,
    max_length: int,
    template: DiffusionTemplate = DEFAULT_TEMPLATE,
) -> TokenizedExample | None:
    prompt = template.prompt_text(instruction)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(response.strip(), add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    answer_ids.append(tokenizer.eos_token_id)
    if not answer_ids or len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return TokenizedExample(prompt_ids + answer_ids, len(prompt_ids))


def corrupt_answer(
    example: TokenizedExample,
    mask_token_id: int,
    rng: random.Random,
    min_mask_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(example.input_ids, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    answer_positions = list(range(example.answer_start, ids.numel()))
    if not answer_positions:
        raise ValueError("example has no answer tokens")

    ratio = rng.uniform(min_mask_ratio, 1.0)
    selected = [position for position in answer_positions if rng.random() < ratio]
    if not selected:
        selected = [rng.choice(answer_positions)]
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    labels[selected_tensor] = ids[selected_tensor]
    ids[selected_tensor] = mask_token_id
    return ids, labels


def iter_jsonl(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("instruction"), str) or not isinstance(record.get("response"), str):
                raise ValueError(f"invalid record at {path}:{line_number}")
            yield record


class DiffusionJsonlDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int):
        self.examples: list[TokenizedExample] = []
        for record in iter_jsonl(path):
            encoded = encode_example(
                tokenizer,
                record["instruction"],
                record["response"],
                max_length,
            )
            if encoded is not None:
                self.examples.append(encoded)
        if not self.examples:
            raise ValueError(f"no usable examples in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.examples[index]


class DiffusionCollator:
    def __init__(self, tokenizer, seed: int, min_mask_ratio: float = 0.15):
        if tokenizer.mask_token_id is None:
            raise ValueError("tokenizer must define mask_token_id")
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.rng = random.Random(seed)
        self.min_mask_ratio = min_mask_ratio

    def __call__(self, examples: list[TokenizedExample]) -> dict[str, torch.Tensor]:
        corrupted = [
            corrupt_answer(example, self.mask_token_id, self.rng, self.min_mask_ratio)
            for example in examples
        ]
        max_length = max(ids.numel() for ids, _ in corrupted)
        input_ids = torch.full((len(examples), max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(examples), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
        for row, (ids, row_labels) in enumerate(corrupted):
            length = ids.numel()
            input_ids[row, :length] = ids
            labels[row, :length] = row_labels
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
