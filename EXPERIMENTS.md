# Architecture, datasets, and local training results

This note records the two completed local experiments. It separates measured results
from architectural limits and preserves the exact conditions needed to interpret the
numbers.

## Model design

The source checkpoint is
[`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2),
an encoder-only ModernBERT model. The conversion retains that backbone and loads it as
`ModernBertForMaskedLM`:

- 12 bidirectional transformer layers
- 384 hidden dimensions
- 12 attention heads
- 1,536-dimensional feed-forward blocks
- 180,000-token vocabulary
- 32,768-token architectural context limit
- 97,768,992 parameters after adding the native masked-LM projection and bias
- input embeddings and output decoder weights tied together

This is not an architectural clone of
[`LiquidAI/LFM2.5-Encoder-350M-Diffusion`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Diffusion).
It applies the same broad masked-diffusion idea to Granite's smaller ModernBERT
backbone.

### Training objective

Every example is formatted as:

```text
Instruction:
<user instruction>

Response:
<answer>
```

The instruction remains visible. For each presentation, a uniformly selected masking
ratio between 15% and 100% corrupts answer tokens. Cross-entropy is calculated only at
the corrupted answer positions. The encoder can therefore use context on both sides of
a mask rather than being restricted to left-to-right attention.

### Generation

Generation appends a fixed block of mask tokens after the prompt. At each iteration the
model predicts every remaining mask, chooses the single position with the most
confident token, fills it, and repeats. The fill order is not left-to-right.

The hard slot ceiling is:

```text
32768 - encoded prompt length - formatting/special tokens
```

That hard limit is not a quality claim. The long run used at most 256 total tokens, and
12 generated slots remain the practical default. A 24-slot probe was grammatical but
repetitive. Decoding also requires one model forward pass per filled slot.

## Dataset

Both runs use the Apache-2.0
[`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend)
conversation dataset. The preparation script takes the first adjacent `human` / `gpt`
pair from each conversation, tokenizes it with the Granite tokenizer, rejects empty or
out-of-window examples, shuffles deterministically with seed `20260824`, and reserves a
held-out split.

| Run | Source shards | Maximum total length | Train | Validation | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial conversion | 1 of 6 | 96 | 7,000 | 256 | 7,308 in shard |
| 10-hour continuation | 6 of 6 | 256 | 498,601 | 1,024 | 499,625 total |

Raw Parquet occupied approximately 1.4 GB locally. Prepared full-run JSONL occupied
approximately 337 MB. Neither data nor generated checkpoints are committed to Git.

## Run 1: initial conversion

| Setting | Value |
| --- | --- |
| Starting model | IBM Granite embedding 97M multilingual R2 |
| Epochs / updates | 3 / 2,625 |
| Microbatch / accumulation | 8 / 1 |
| Learning rate | `5e-5` backbone, `5e-4` new MLM components |
| Head-only warm start | 100 updates |
| Precision | BF16 |
| Wall time | 333.4 seconds |
| Peak PyTorch allocated VRAM | 1,526.5 MiB |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB advertised |

Evaluation on this run's 256-example held-out split:

| Measurement | Result |
| --- | ---: |
| Random-mask cross-entropy | 3.6605 |
| Random-mask token accuracy | 42.62% |
| Fully masked-answer cross-entropy | 5.4220 |
| Fully masked-answer token accuracy | 15.80% |

This established that the encoder could be converted and generate grammatical text,
but generations were repetitive and frequently incorrect.

## Run 2: 10-hour continuation

| Setting | Value |
| --- | --- |
| Starting model | Final checkpoint from run 1 |
| Time cap / measured wall time | 10 hours / 36,000.8 seconds |
| Optimizer updates | 226,738 |
| Example presentations | 1,360,418 |
| Effective dataset passes | 2.7285 |
| Microbatch / accumulation | 6 / 1 |
| Learning rate | `1e-5` for all parameters |
| Scheduler / warmup | Constant / 200 updates |
| Maximum total sequence | 256 tokens |
| Precision | BF16 |
| Timed recovery saves | 2, 4, 6, and 8 hours |
| Peak PyTorch allocated VRAM | 2,811.0 MiB |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB advertised |

The trainer stopped itself with `stop_reason=duration_complete`; it did not stop due to
an out-of-memory condition or error.

### Fair before/after evaluation

Both checkpoints were evaluated with the same seed on the same 1,024-example held-out
split from the 256-token dataset:

| Measurement | Before continuation | After 10 hours |
| --- | ---: | ---: |
| Random-mask cross-entropy | 3.9989 | 2.1140 |
| Random-mask token accuracy | 36.88% | 58.81% |
| Fully masked-answer cross-entropy | 6.1450 | 4.6621 |
| Fully masked-answer token accuracy | 9.01% | 16.27% |

Lower reconstruction loss produced visibly better short-answer fluency, but did not
produce a reliable factual assistant. Fixed probes still failed `7 x 8`, the capital
of France, and English-to-Hindi translation. These failures are part of the result and
should not be hidden behind aggregate token accuracy.

## Artifact boundary

The final safetensors file is 391,083,856 bytes and has SHA-256:

```text
e3361bdbd87c22a5ed3bdf778e0063cda02f070d45f75e6d61cc62be03219d08
```

The checkpoint, intermediate recovery saves, downloaded Parquet, prepared JSONL, venv,
and caches remain local because they are large generated artifacts. This repository
contains the code, exact commands, tests, consolidated metrics, and documentation
needed to reproduce them.
