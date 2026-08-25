#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from granite_diffusion.data import DiffusionCollator, DiffusionJsonlDataset
from granite_diffusion.generation import generate_diffusion
from granite_diffusion.modeling import load_checkpoint


DEFAULT_PROMPTS = [
    "Give one short tip for writing clearer code.",
    "Write one sentence explaining why plants need sunlight.",
    "What is 7 multiplied by 8? Answer briefly.",
    "Translate 'Good morning' to Hindi.",
    "Name the capital of France and its country in one sentence.",
]


def evaluate_reconstruction(model, tokenizer, validation_file: str, max_length: int, mask_ratio: float):
    dataset = DiffusionJsonlDataset(validation_file, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=DiffusionCollator(tokenizer, seed=314159, min_mask_ratio=mask_ratio),
        num_workers=0,
    )
    loss_sum = 0.0
    correct = 0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                outputs = model(**batch)
            targets = batch["labels"][batch["labels"] != -100]
            count = targets.numel()
            loss_sum += outputs.loss.float().item() * count
            correct += (outputs.logits.argmax(dim=-1) == targets).sum().item()
            token_count += count
    loss = loss_sum / token_count
    return {
        "examples": len(dataset),
        "masked_tokens": token_count,
        "cross_entropy": loss,
        "token_accuracy": correct / token_count,
        "exp_cross_entropy": math.exp(min(loss, 20)),
        "minimum_mask_ratio": mask_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out diffusion reconstruction and generation.")
    parser.add_argument("--model", default="outputs/granite-diffusion-97m/final")
    parser.add_argument("--validation-file", default="data/processed/validation.jsonl")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", default="outputs/granite-diffusion-97m/evaluation.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.model, device)
    metrics = evaluate_reconstruction(model, tokenizer, args.validation_file, args.max_length, 0.15)
    full_mask_metrics = evaluate_reconstruction(model, tokenizer, args.validation_file, args.max_length, 1.0)
    generations = []
    for prompt in DEFAULT_PROMPTS:
        result = generate_diffusion(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        generations.append({"prompt": prompt, "text": result.text, "fill_order": result.fill_order})
    report = {
        "model": args.model,
        "device": str(device),
        "held_out_random_noise": metrics,
        "held_out_full_mask": full_mask_metrics,
        "generations": generations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
