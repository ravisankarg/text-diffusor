#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from granite_diffusion.formatting import DEFAULT_TEMPLATE
from granite_diffusion.modeling import BASE_MODEL_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Create a bounded local SFT subset from open-perfectblend.")
    parser.add_argument("--parquet", required=True, nargs="+")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=0, help="0 keeps every eligible example except validation")
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def extract_pair(conversation):
    for index in range(len(conversation) - 1):
        first, second = conversation[index], conversation[index + 1]
        if first.get("from") == "human" and second.get("from") == "gpt":
            instruction = (first.get("value") or "").strip()
            response = (second.get("value") or "").strip()
            if instruction and response:
                return instruction, response
    return None


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    accepted = []
    for parquet_path in args.parquet:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["conversations", "source"]):
            candidates = []
            for row in batch.to_pylist():
                pair = extract_pair(row["conversations"] or [])
                if pair is None:
                    continue
                candidates.append((pair[0], pair[1], row["source"]))
            if not candidates:
                continue
            prompts = [DEFAULT_TEMPLATE.prompt_text(pair[0]) for pair in candidates]
            responses = [pair[1] for pair in candidates]
            prompt_lengths = tokenizer(prompts, add_special_tokens=True, return_length=True)["length"]
            response_lengths = tokenizer(responses, add_special_tokens=False, return_length=True)["length"]
            for pair, prompt_length, response_length in zip(candidates, prompt_lengths, response_lengths):
                total_length = prompt_length + response_length + 1  # EOS
                if total_length < 12 or total_length > args.max_length:
                    continue
                accepted.append({"instruction": pair[0], "response": pair[1], "source": pair[2]})

    train_size = args.train_size or len(accepted) - args.validation_size
    needed = train_size + args.validation_size
    if len(accepted) < needed:
        raise RuntimeError(f"only {len(accepted)} examples fit max_length={args.max_length}; need {needed}")
    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    selected = accepted[:needed]
    validation = selected[: args.validation_size]
    train = selected[args.validation_size : args.validation_size + train_size]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train), ("validation", validation)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "source_dataset": "mlabonne/open-perfectblend",
        "source_files": [Path(path).name for path in args.parquet],
        "model_tokenizer": args.model,
        "max_length": args.max_length,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "eligible_examples": len(accepted),
        "seed": args.seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
