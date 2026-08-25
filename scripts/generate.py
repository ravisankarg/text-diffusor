#!/usr/bin/env python3
import argparse
import json

import torch

from granite_diffusion.generation import generate_diffusion
from granite_diffusion.modeling import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text by iterative masked diffusion.")
    parser.add_argument("--model", default="outputs/granite-diffusion-97m/final")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.model, device)
    result = generate_diffusion(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps({"text": result.text, "fill_order": result.fill_order}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
