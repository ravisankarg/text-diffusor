# Granite Diffusion 97M

This project converts `ibm-granite/granite-embedding-97m-multilingual-r2` from an
encoder-only embedding checkpoint into a masked-diffusion text generator. It follows
the mechanism demonstrated by `LiquidAI/LFM2.5-Encoder-350M-Diffusion`: train a
bidirectional masked language model on prompt/answer pairs, begin generation with a
block of mask tokens, and iteratively fill the most confident position.

The result is a converted 97M-parameter ModernBERT model, not an architectural copy of
LFM2.5. Granite remains ModernBERT; the conversion adds its native tied masked-LM head
and instruction diffusion objective. Model quality depends on the amount of local SFT.

## Why it fits a 4 GB GPU

- The 180,000-token output decoder is tied to Granite's existing input embeddings.
- ModernBERT sparse prediction computes vocabulary logits only at corrupted positions.
- Training supports FP16 scaling and activation checkpointing. The validated run fit
  directly with BF16, Adafactor, and microbatch 8, so checkpointing was unnecessary.
- `scripts/train.py` refuses to train without CUDA and records actual peak allocated VRAM.

## Related design direction

[MULTILINGUAL_CONTENT_CLASSIFIER.md](MULTILINGUAL_CONTENT_CLASSIFIER.md) records a
proposed non-generative use of the Granite encoder: one-pass, multilingual domain,
content-kind, and facet classification for on-device search. It covers the product
rationale, QP/search integration, dataset strategy, OCR robustness, training plan,
and evaluation gates. This is a design direction, not an implemented or benchmarked
result of the diffusion checkpoint.

## Reproduce

All Python packages live in `.venv`.

The helper script creates the requested virtual environment and installs the pinned
dependencies:

```bash
bash scripts/setup_venv.sh
```

Download all six source-data shards:

```bash
bash scripts/download_open_perfectblend.sh
```

The complete measured architecture, dataset filtering, both training runs, evaluation
tables, and limitations are recorded in [EXPERIMENTS.md](EXPERIMENTS.md). Consolidated
machine-readable measurements are in [results/metrics.json](results/metrics.json).

The equivalent manual environment setup is:

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

Prepare a bounded local subset from the first Apache-2.0 `open-perfectblend` Parquet
shard:

```bash
.venv/bin/python scripts/prepare_data.py \
  --parquet data/raw/data/train-00000-of-00006.parquet \
  --train-size 7000 --validation-size 256 --max-length 96
```

Train on the local GPU:

```bash
.venv/bin/python scripts/train.py \
  --max-length 96 --micro-batch-size 8 --gradient-accumulation 1 \
  --epochs 3 --head-only-steps 100 --no-gradient-checkpointing
```

Generate by iterative unmasking:

```bash
.venv/bin/python scripts/generate.py \
  --model outputs/granite-diffusion-97m/final \
  --prompt "Give one short tip for writing clearer code." \
  --max-new-tokens 12
```

Evaluate held-out masked reconstruction and five fixed unseen prompts:

```bash
.venv/bin/python scripts/evaluate.py
```

Exact wrappers for both measured runs are provided as `scripts/train_initial.sh` and
`scripts/train_10h.sh`. The latter preserves the first checkpoint, saves recovery
checkpoints every two hours, and stops automatically after ten hours.

## Validated local result: initial conversion

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB advertised
- Training: 7,000 examples, 3 epochs, 2,625 optimizer updates, BF16
- Wall time: 333.4 seconds
- Peak PyTorch allocated VRAM: 1,526.5 MiB
- Held-out random-noise loss / token accuracy: 3.6605 / 42.62%
- Held-out all-masked-answer loss / token accuracy: 5.4220 / 15.80%
- Final checkpoint: 97,768,992 parameters, tied decoder, 391 MB FP32 safetensors

The model genuinely generates text through iterative unmasking. The short local run is
not a strong factual assistant: validated examples are grammatical but can repeat or
miss simple facts and arithmetic. Twelve generated tokens are noticeably more stable
than 24 and are therefore the default.

## Validated local result: 10-hour continuation

The initial checkpoint was continued on all eligible conversations from the six
`open-perfectblend` shards. Preparation produced 498,601 training examples and 1,024
held-out examples at a 256-token maximum sequence length. The source checkpoint was
preserved; the continued checkpoint is in
`outputs/granite-diffusion-97m-10h/final`.

```bash
.venv/bin/python scripts/train.py \
  --model outputs/granite-diffusion-97m/final \
  --train-file data/processed-256/train.jsonl \
  --validation-file data/processed-256/validation.jsonl \
  --output-dir outputs/granite-diffusion-97m-10h \
  --max-length 256 --epochs 1000 \
  --micro-batch-size 6 --gradient-accumulation 1 \
  --head-only-steps 0 --learning-rate 1e-5 --head-learning-rate 1e-5 \
  --warmup-steps 200 --scheduler constant \
  --save-steps 0 --save-minutes 120 --max-duration-hours 10 \
  --log-steps 100 --no-gradient-checkpointing
```

- Duration: 36,000.8 seconds, stopped automatically at the requested 10 hours
- Work completed: 226,738 updates / 1,360,418 example presentations, or 2.73 passes
- Peak PyTorch allocated VRAM: 2,811.0 MiB
- Held-out random-mask loss / token accuracy: 2.1140 / 58.81%
- Held-out fully-masked-answer loss / token accuracy: 4.6621 / 16.27%
- Matching pre-continuation baseline: 3.9989 / 36.88% and 6.1450 / 9.01%
- Recovery checkpoints: 2, 4, 6, and 8 hours

The continuation materially improves held-out token reconstruction and short-answer
fluency. It still fails simple factual, arithmetic, and translation probes, so it is an
experimental diffusion language model rather than a reliable assistant.

## Generation-slot limits

ModernBERT's architectural context limit is 32,768 total tokens. The hard upper bound
for generation slots is therefore `32768 - encoded prompt length - special/formatting
tokens`. That is only a structural limit: this run trained with a 256-token total
sequence window, and the validated practical default remains 12 slots. Very long slot
blocks are outside the training distribution, require one or more forward passes per
unmasking step, and should not be expected to produce useful text.

## Reference boundary

Liquid AI reports that its released diffusion checkpoint is a full fine-tune of its
masked-LM encoder on roughly 1.39M conversations for three epochs. This repository uses
the same generation principle on 498,601 locally eligible examples and completed 2.73
passes in ten hours on a 4 GB laptop GPU. Do not equate matching a wall-clock budget
with the capability or training coverage of Liquid AI's released model.

References:

- https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2
- https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Diffusion
- https://huggingface.co/datasets/mlabonne/open-perfectblend
