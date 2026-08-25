from dataclasses import dataclass

import torch

from .formatting import DEFAULT_TEMPLATE, DiffusionTemplate


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    fill_order: list[int]


def _masked_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, positions: torch.Tensor):
    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden = outputs.last_hidden_state[0, positions]
    return model.decoder(model.head(hidden))


@torch.inference_mode()
def generate_diffusion(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 12,
    temperature: float = 0.0,
    template: DiffusionTemplate = DEFAULT_TEMPLATE,
) -> GenerationResult:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids = tokenizer.encode(template.prompt_text(instruction), add_special_tokens=True)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    requested_length = len(prompt_ids) + max_new_tokens
    if context_limit is not None and requested_length > context_limit:
        available_slots = max(0, context_limit - len(prompt_ids))
        raise ValueError(
            f"prompt uses {len(prompt_ids)} tokens, so max_new_tokens cannot exceed "
            f"{available_slots} for this model's {context_limit}-token context"
        )
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [prompt_ids + [tokenizer.mask_token_id] * max_new_tokens],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    fill_order: list[int] = []
    special_ids = set(tokenizer.all_special_ids) - {tokenizer.eos_token_id}

    model.eval()
    for _ in range(max_new_tokens):
        positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            break
        logits = _masked_logits(model, input_ids, attention_mask, positions).float()
        for token_id in special_ids:
            if 0 <= token_id < logits.shape[-1]:
                logits[:, token_id] = -torch.inf
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            candidates = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            confidence = probabilities.gather(1, candidates[:, None]).squeeze(1)
        else:
            probabilities = torch.softmax(logits, dim=-1)
            confidence, candidates = probabilities.max(dim=-1)
        best = confidence.argmax()
        absolute_position = positions[best]
        input_ids[0, absolute_position] = candidates[best]
        fill_order.append(int(absolute_position) - len(prompt_ids))

    generated = input_ids[0, len(prompt_ids):].tolist()
    if tokenizer.eos_token_id in generated:
        generated = generated[: generated.index(tokenizer.eos_token_id)]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return GenerationResult(text=text, token_ids=generated, fill_order=fill_order)
