"""Post-training weight quantization: store weights in 8 or 4 bits.

A trained model's weights are stored as 32- or 64-bit floats, but LLMs are
usually served with 8- or 4-bit weights (LLM.int8, GPTQ, llama.cpp's
Q4/Q8 formats): memory, and the time to read the weights, shrink 4-8x.
The simplest scheme, used here, is symmetric round-to-nearest in groups:

    for every group of G consecutive weights in a row:
        scale = max |w| / (2^(bits-1) - 1)          (127 for 8 bits, 7 for 4)
        code  = round(w / scale)                     an integer in [-127, 127] or [-7, 7]
        w'    = code * scale                         what inference computes with

Only the codes and one scale per group are stored. Small groups follow the
local size of the weights more closely (less error) but store more scales.

This module quantizes every weight matrix (linear layers and embeddings;
biases and norm scales stay as they are) and measures what it costs in
validation loss on a trained character model:

    python3 quantize.py runs/baby_llama --bits 8 4 --group-size 32
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensor import Tensor
from tensor_nn import TensorModule


def quantize(values, *, bits: int, group_size: int) -> tuple[list[int], list[float]]:
    """Integer codes (one per value) and one scale per group of ``group_size``."""

    if bits < 2:
        raise ValueError("bits must be at least 2")
    levels = 2 ** (bits - 1) - 1
    codes, scales = [], []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        scale = max(abs(value) for value in group) / levels or 1.0      # an all-zero group keeps scale 1
        scales.append(scale)
        codes += [max(-levels, min(levels, round(value / scale))) for value in group]
    return codes, scales


def dequantize(codes: list[int], scales: list[float], *, group_size: int) -> list[float]:
    return [code * scales[index // group_size] for index, code in enumerate(codes)]


def quantized_matrices(model: TensorModule) -> list[Tensor]:
    """The weights quantization applies to: every parameter with 2+ dimensions."""

    return [parameter for parameter in model.parameters() if parameter.ndim >= 2]


def quantize_model(model: TensorModule, *, bits: int, group_size: int = 32) -> dict:
    """Replace every weight matrix by its quantized-then-dequantized values
    (rows are cut into groups, so a group never spans two rows). Returns the
    weights' storage size before (float32) and after (codes + float32 scales)."""

    count = scales = 0
    for weight in quantized_matrices(model):
        width = weight.shape[-1]
        values = list(weight.data)
        restored = []
        for row in range(0, len(values), width):
            codes, row_scales = quantize(values[row : row + width], bits=bits, group_size=group_size)
            restored += dequantize(codes, row_scales, group_size=group_size)
            scales += len(row_scales)
        weight.data = restored
        count += len(values)
    return {"weights": count, "float32_bytes": 4 * count, "quantized_bytes": count * bits / 8 + 4 * scales}


# ---------------------------------------------------------------------------
# Measuring it on a trained character model (a gpt_train.py run directory)
# ---------------------------------------------------------------------------


def load_run(run_dir: str | Path):
    from char_checkpoint import load_checkpoint
    from gpt_train import build_model, split_text
    from text import CharVocab

    run_dir = Path(run_dir)
    config = argparse.Namespace(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    config.kv_heads = getattr(config, "kv_heads", None)
    config.experts = getattr(config, "experts", 0)
    config.expert_top_k = getattr(config, "expert_top_k", None) or 2
    config.window = getattr(config, "window", None)
    text = Path(config.text_file).read_text(encoding="utf-8")
    vocab = CharVocab.from_text(text)
    model = build_model(len(vocab), config)
    load_checkpoint(run_dir / "model.json", model, vocab, model_name="transformer")
    _, validation = split_text(text, config.validation_fraction)
    return model, vocab.encode(validation), config


def validation_loss(model, encoded: list[int], context_size: int, *, batch_size: int = 64) -> float:
    """Mean next-character loss over the whole validation text, read in
    consecutive non-overlapping windows (the same every time)."""

    from parallel import sequence_loss
    from tensor import no_grad
    from train import evaluation_mode

    starts = list(range(0, len(encoded) - context_size - 1, context_size))
    total = 0.0
    for first in range(0, len(starts), batch_size):
        chunk = starts[first : first + batch_size]
        xs = [encoded[start : start + context_size] for start in chunk]
        ys = [encoded[start + 1 : start + context_size + 1] for start in chunk]
        with no_grad(), evaluation_mode(model):
            total += sequence_loss(model, xs, ys).item() * len(chunk)
    return total / len(starts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="a gpt_train.py --out-dir of a character transformer")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args(argv)

    model, encoded, config = load_run(args.run_dir)
    original = [list(parameter.data) for parameter in model.parameters()]
    base = validation_loss(model, encoded, config.context_size)
    print(f"{'weights':>14s}  {'size':>9s}  {'val loss':>8s}  {'bits/char':>9s}  change")
    print(f"{'float32':>14s}  {4 * sum(w.numel for w in quantized_matrices(model)) / 1024:7.0f} KB  "
          f"{base:8.4f}  {base / math.log(2):9.4f}")
    for bits in args.bits:
        for parameter, values in zip(model.parameters(), original):
            parameter.data = values
        sizes = quantize_model(model, bits=bits, group_size=args.group_size)
        loss = validation_loss(model, encoded, config.context_size)
        print(f"{f'int{bits}, groups of {args.group_size}':>14s}  {sizes['quantized_bytes'] / 1024:7.0f} KB  "
              f"{loss:8.4f}  {loss / math.log(2):9.4f}  {(loss - base) / math.log(2):+.4f}", flush=True)


if __name__ == "__main__":
    main()
