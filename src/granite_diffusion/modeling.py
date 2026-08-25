from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, ModernBertForMaskedLM


BASE_MODEL_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"


def _initialize_missing_mlm_head(model: ModernBertForMaskedLM) -> None:
    """Start the missing projection close to an identity mapping."""
    with torch.no_grad():
        model.head.dense.weight.zero_()
        model.head.dense.weight.fill_diagonal_(1.0)
        model.head.norm.weight.fill_(1.0)
        if model.decoder.bias is not None:
            model.decoder.bias.zero_()


def load_base_for_diffusion(model_name_or_path: str = BASE_MODEL_ID):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.architectures = ["ModernBertForMaskedLM"]
    config.sparse_prediction = True
    config.repad_logits_with_grad = False
    config.gradient_checkpointing = True
    config.diffusion_objective = "answer_only_random_mask"
    config.diffusion_format = "Instruction/Response"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model, loading = ModernBertForMaskedLM.from_pretrained(
        model_name_or_path,
        config=config,
        dtype=torch.float32,
        output_loading_info=True,
    )
    expected_missing = {"decoder.bias", "head.dense.weight", "head.norm.weight"}
    unexpected_missing = set(loading["missing_keys"]) - expected_missing
    if unexpected_missing:
        raise RuntimeError(f"unexpected missing weights: {sorted(unexpected_missing)}")
    if loading["unexpected_keys"]:
        raise RuntimeError(f"unexpected checkpoint weights: {loading['unexpected_keys']}")
    _initialize_missing_mlm_head(model)
    model.tie_weights()
    return model, tokenizer


def load_checkpoint(checkpoint: str | Path, device: str | torch.device = "cpu"):
    checkpoint = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = ModernBertForMaskedLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.to(device)
    return model, tokenizer
