"""Utility helpers to instantiate and load the DeepSeek-OCR MLX model."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
import mlx.core as mx

from .config import DeepSeekOCRConfig
from .model import DeepSeekOCRForCausalLM


def load_config(model_path: Path) -> DeepSeekOCRConfig:
    """Load a Hugging Face `config.json` into :class:`DeepSeekOCRConfig`."""

    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return DeepSeekOCRConfig.from_dict(raw)


def _gather_weights(model_path: Path) -> dict[str, mx.array]:
    """Read all `.safetensors` shards under ``model_path``."""

    weights: dict[str, mx.array] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        loaded = mx.load(str(shard))
        if not isinstance(loaded, dict):
            raise ValueError(f"Unexpected payload type for {shard}: {type(loaded)!r}")
        weights.update(loaded)
    if not weights:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    return weights


def load_model(model_path: Path, *, lazy: bool = False) -> DeepSeekOCRForCausalLM:
    """Instantiate ``DeepSeekOCRForCausalLM`` and load MLX weights."""

    config = load_config(model_path)
    model = DeepSeekOCRForCausalLM(config)

    weights = _gather_weights(model_path)
    weights = model.model.sanitize(weights)

    casted: dict[str, mx.array] = {}
    for name, tensor in weights.items():
        if tensor.dtype == mx.float32:
            casted[name] = tensor.astype(mx.float16)
        else:
            casted[name] = tensor

    weights = casted

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(model_path: Path | str, *, lazy: bool = False) -> DeepSeekOCRForCausalLM:
    """Convenience wrapper accepting either a :class:`Path` or string."""

    resolved = Path(model_path)
    return load_model(resolved, lazy=lazy)


def load_with_tokenizer(
    model_path: Path | str | None = None, device_number: int | None = None,
    *, model_transform: Callable[[DeepSeekOCRForCausalLM], Any] | None = None,
    model_id: str = "deepseek-ai/DeepSeek-OCR",
) -> tuple[Any, DeepSeekOCRForCausalLM]:
    """Load one local Metal model/tokenizer pair, applying a model hook once.

    Accept a checkpoint, a Hugging Face cache root, or ``None`` for the default
    cache. Resolve cached snapshots on each call without downloading files.
    Device aliases and cache/checkpoint paths share one resolved cache entry;
    changing the checkpoint or transform replaces the loaded pair.
    """
    if device_number not in (None, 0):
        raise ValueError("MLX OCR supports only the default Metal device (None or 0)")
    return _load_with_tokenizer(_cached_checkpoint(model_path, model_id), model_transform)


def _cached_checkpoint(model_path: Path | str | None, model_id: str) -> Path:
    from huggingface_hub import constants, try_to_load_from_cache

    root = Path(model_path if model_path is not None else constants.HF_HUB_CACHE).expanduser().resolve()
    if (root / "config.json").is_file():
        return root
    cached = try_to_load_from_cache(model_id, "config.json", cache_dir=root)
    if isinstance(cached, str):
        return Path(cached).parent.resolve()
    snapshots = root / ("models--" + model_id.replace("/", "--")) / "snapshots"
    config = max(snapshots.glob("*/config.json"), key=lambda path: path.parent.stat().st_mtime, default=None)
    if config is None:
        raise FileNotFoundError(f"No cached {model_id} checkpoint in {root}; download the model first")
    return config.parent.resolve()


@lru_cache(maxsize=1)
def _load_with_tokenizer(model_path: Path, model_transform: Callable | None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = load(model_path)
    if model_transform is not None:
        model = model_transform(model)
    return tokenizer, model
