#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from granite_diffusion.data import DiffusionCollator, DiffusionJsonlDataset
from granite_diffusion.generation import generate_diffusion
from granite_diffusion.modeling import BASE_MODEL_ID, load_base_for_diffusion, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Granite 97M as a masked-diffusion generator.")
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--validation-file", default="data/processed/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/granite-diffusion-97m")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--head-only-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-minutes", type=float, default=0.0)
    parser.add_argument("--max-duration-hours", type=float, default=0.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_base_trainable(model, trainable: bool):
    for parameter in model.model.parameters():
        parameter.requires_grad = trainable
    # decoder.weight is tied to model embeddings and inherits the same flag.
    model.head.requires_grad_(True)
    if model.decoder.bias is not None:
        model.decoder.bias.requires_grad_(True)


def save_model(model, tokenizer, output_dir: Path, training_state: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_state.json").write_text(json.dumps(training_state, indent=2) + "\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required: training is intentionally local-GPU only")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if Path(args.model).is_dir():
        model, tokenizer = load_checkpoint(args.model)
        model.config.sparse_prediction = True
        model.sparse_prediction = True
        continued_from = str(Path(args.model).resolve())
    else:
        model, tokenizer = load_base_for_diffusion(args.model)
        continued_from = None
    model.config.diffusion_training_max_length = args.max_length
    model.config.diffusion_default_generation_tokens = 12
    use_checkpointing = not args.no_gradient_checkpointing
    if use_checkpointing and args.head_only_steps == 0:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.to("cuda")

    train_dataset = DiffusionJsonlDataset(args.train_file, tokenizer, args.max_length)
    validation_dataset = DiffusionJsonlDataset(args.validation_file, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=DiffusionCollator(tokenizer, args.seed),
        num_workers=0,
    )

    new_parameters = list(model.head.parameters()) + ([model.decoder.bias] if model.decoder.bias is not None else [])
    new_ids = {id(parameter) for parameter in new_parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in new_ids]
    optimizer = Adafactor(
        [
            {"params": base_parameters, "lr": args.learning_rate},
            {"params": new_parameters, "lr": args.head_learning_rate},
        ],
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    set_base_trainable(model, args.head_only_steps == 0)
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    example_presentations = 0
    running_loss = 0.0
    started = time.monotonic()
    deadline = started + args.max_duration_hours * 3600 if args.max_duration_hours else None
    next_timed_save = started + args.save_minutes * 60 if args.save_minutes else None
    stop_reason = "epochs_or_steps_complete"
    epochs_started = 0
    model.train()

    for epoch in range(args.epochs):
        epochs_started = epoch + 1
        for batch in train_loader:
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "duration_complete"
                break
            if update_step >= total_steps:
                break
            if update_step == args.head_only_steps and args.head_only_steps > 0:
                set_base_trainable(model, True)
                if use_checkpointing:
                    model.gradient_checkpointing_enable()
                print(f"step={update_step} unfreezing Granite encoder")
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            example_presentations += batch["input_ids"].shape[0]
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss = model(**batch).loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            # `loss` is already divided by accumulation. Summing microbatches
            # therefore gives one mean unscaled loss per optimizer update.
            running_loss += loss.item()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = not scaler.is_enabled() or scaler.get_scale() >= scale_before
            if not optimizer_ran:
                optimizer.zero_grad(set_to_none=True)
                print("gradient overflow: skipped optimizer update", flush=True)
                continue
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if update_step % args.log_steps == 0:
                elapsed = time.monotonic() - started
                peak = torch.cuda.max_memory_allocated() / 2**20
                print(
                    f"step={update_step}/{total_steps} loss={running_loss / args.log_steps:.4f} "
                    f"elapsed_hours={elapsed / 3600:.3f} updates_per_s={update_step / elapsed:.3f} "
                    f"peak_vram_mib={peak:.0f}",
                    flush=True,
                )
                running_loss = 0.0

            if args.save_steps and update_step % args.save_steps == 0:
                state = {"step": update_step, "epoch": epoch, "elapsed_seconds": time.monotonic() - started,
                         "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20}
                save_model(model, tokenizer, output_dir / f"checkpoint-{update_step}", state)
            if next_timed_save is not None and time.monotonic() >= next_timed_save:
                state = {"step": update_step, "epoch": epoch, "elapsed_seconds": time.monotonic() - started,
                         "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20}
                save_model(model, tokenizer, output_dir / f"checkpoint-{update_step}", state)
                next_timed_save += args.save_minutes * 60

        if stop_reason == "duration_complete" or update_step >= total_steps:
            break

    final_state = {
        "step": update_step,
        "epochs_started": epochs_started,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "example_presentations": example_presentations,
        "dataset_passes": example_presentations / len(train_dataset),
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0),
        "precision": precision,
        "continued_from": continued_from,
        "requested_duration_hours": args.max_duration_hours,
        "stop_reason": stop_reason,
    }
    save_model(model, tokenizer, output_dir / "final", final_state)
    sample = generate_diffusion(model, tokenizer, "Give one short tip for writing clearer code.", max_new_tokens=16)
    final_state["sample"] = sample.text
    (output_dir / "final" / "training_state.json").write_text(json.dumps(final_state, indent=2) + "\n")
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
