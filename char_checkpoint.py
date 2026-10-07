"""Character-model checkpoints: model state plus the vocabulary and config.

A v1 checkpoint is one JSON file:

    {
      "format": "nanollm.char_demo.v1",
      "model": "transformer",          # bigram | embedding | transformer
      "vocab": ["\\n", " ", "a", ...],    # index -> character
      "state": {...}                    # model.state_dict(), incl. architecture
    }

Saving the vocabulary and architecture means a checkpoint can be evaluated or
sampled later without repeating the training flags or even the training text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensor_nn import TensorModule
from text import CharVocab


CHECKPOINT_FORMAT = "nanollm.char_demo.v1"


def save_checkpoint(
    path: str | Path,
    model: TensorModule,
    vocab: CharVocab,
    *,
    model_name: str,
) -> None:
    """Save model state plus the vocabulary needed to interpret logits."""

    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": model_name,
        "vocab": vocab.itos,
        "state": model.state_dict(),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(
    path: str | Path,
    model: TensorModule,
    vocab: CharVocab,
    *,
    model_name: str,
) -> None:
    """Load a char-demo checkpoint, accepting older raw state files."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "format" not in payload:
        model.load_state_dict(payload)
        return
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported char checkpoint format")
    if payload.get("model") != model_name:
        raise ValueError("checkpoint model does not match requested char model")
    if payload.get("vocab") != vocab.itos:
        raise ValueError("checkpoint vocabulary does not match text vocabulary")
    model.load_state_dict(payload["state"])


def apply_checkpoint_config(args: argparse.Namespace, path: str | Path) -> None:
    """Fill unspecified CLI model settings from a v1 checkpoint."""

    payload = _checkpoint_payload(path)
    if "format" not in payload:
        return
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported char checkpoint format")

    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a dictionary")

    updates = _checkpoint_config_updates(payload.get("model"), state)
    missing = [
        name
        for name, value in updates.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "checkpoint is missing model config: " + ", ".join(missing)
        )

    explicit_options = getattr(args, "_explicit_options", set())
    for name, value in updates.items():
        if name not in explicit_options:
            setattr(args, name, value)


def checkpoint_vocab(path: str | Path) -> CharVocab:
    """Load the saved character vocabulary from a v1 checkpoint."""

    payload = _checkpoint_payload(path)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("generate-only requires a v1 char checkpoint")
    vocab = payload.get("vocab")
    if not isinstance(vocab, list):
        raise ValueError("checkpoint vocabulary must be a list")
    return CharVocab(vocab)


def _checkpoint_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_config_updates(model_name, state: dict) -> dict:
    if model_name == "bigram":
        return {
            "model": "bigram",
            "context_size": state.get("context_size"),
        }
    if model_name == "embedding":
        return {
            "model": "embedding",
            "context_size": state.get("context_size"),
            "embedding_dim": state.get("embedding_dim"),
        }
    if model_name == "transformer":
        return {
            "model": "transformer",
            "context_size": state.get("context_size"),
            "embedding_dim": state.get("embedding_dim"),
            "hidden_dim": state.get("hidden_dim"),
            "heads": state.get("num_heads", 1),
            "layers": state.get("num_layers"),
            "activation": state.get("feed_forward_activation", "relu"),
            "tie_embeddings": state.get("tie_embeddings", False),
        }
    raise ValueError("checkpoint model is unknown")
