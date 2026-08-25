import random
from types import SimpleNamespace

import pytest
import torch

from granite_diffusion.data import TokenizedExample, corrupt_answer
from granite_diffusion.formatting import DiffusionTemplate
from granite_diffusion.generation import generate_diffusion


def test_template_normalizes_instruction_whitespace():
    template = DiffusionTemplate()
    assert template.prompt_text("  explain   gravity\nbriefly ") == "Instruction:\nexplain gravity briefly\n\nResponse:\n"


def test_corruption_changes_only_answer_and_sets_sparse_labels():
    example = TokenizedExample(input_ids=[10, 11, 12, 20, 21, 22], answer_start=3)
    corrupted, labels = corrupt_answer(example, mask_token_id=99, rng=random.Random(7), min_mask_ratio=0.15)
    assert corrupted[:3].tolist() == [10, 11, 12]
    assert labels[:3].tolist() == [-100, -100, -100]
    selected = labels != -100
    assert selected.any()
    assert torch.all(corrupted[selected] == 99)
    assert labels[selected].tolist() == torch.tensor(example.input_ids)[selected].tolist()


def test_full_noise_masks_every_answer_token():
    example = TokenizedExample(input_ids=[1, 2, 3, 4], answer_start=2)
    corrupted, labels = corrupt_answer(example, mask_token_id=9, rng=random.Random(0), min_mask_ratio=1.0)
    assert corrupted.tolist() == [1, 2, 9, 9]
    assert labels.tolist() == [-100, -100, 3, 4]


def test_generation_rejects_slots_beyond_context_limit():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens
            return [1, 2, 3]

    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8))
    with pytest.raises(ValueError, match="cannot exceed 5"):
        generate_diffusion(model, Tokenizer(), "test", max_new_tokens=6)
