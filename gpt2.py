"""GPT-2 small (Radford et al., 2019) in NanoGrad, from OpenAI's published weights.

GPT-2 is the transformer in ``language.py`` at full size: 12 blocks, 12
heads, 768 wide, learned positions over 1,024 tokens, LayerNorm, GELU
(the tanh approximation), and the output layer tied to the token
embedding: 124M parameters. Nothing new is needed but this file, which

- reads the weights (Hugging Face's ``model.safetensors``: a JSON header,
  then the raw float32 numbers) and puts them into the model;
- reads text as GPT-2 does: byte-level BPE with its 50,257 tokens;
- checks the model against PyTorch's (Hugging Face ``transformers``): same
  tokens, same logits, same loss, same gradients;
- samples, and fine-tunes on a text.

    .venv/bin/python gpt2.py --download                       # 550 MB into data/gpt2/
    .venv/bin/python gpt2.py --compare                        # needs torch and transformers
    NANOGRAD_BACKEND=cuda NANOGRAD_DTYPE=float32 .venv/bin/python gpt2.py --sample "The meaning of life is"
    NANOGRAD_BACKEND=cuda NANOGRAD_DTYPE=float32 .venv/bin/python gpt2.py --finetune data/tinyshakespeare/input.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import struct
import urllib.request
from array import array
from pathlib import Path
from time import perf_counter

import kernels
from gpt_train import generate_tokens
from language import CharTransformerModel
from optim import TensorAdam
from parallel import sequence_loss
from tensor import no_grad
from train import evaluation_mode

FILES = ("model.safetensors", "vocab.json", "merges.txt", "config.json")
URL = "https://huggingface.co/openai-community/gpt2/resolve/main/"


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (directory / name).exists():
            print(f"downloading {URL}{name}", flush=True)
            urllib.request.urlretrieve(URL + name, directory / name)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's table from the 256 byte values to printable characters, so
    merges can be stored as text: printable bytes stand for themselves, the
    rest (spaces, control bytes) are moved up past 255."""

    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    table, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(256 + extra)
            extra += 1
    return table


class GPT2Tokenizer:
    """Byte-level BPE: text is cut into words, each word's UTF-8 bytes are
    merged pair by pair in the order of ``merges.txt``.

    GPT-2 cuts words with ``\\p{L}`` (letters) and ``\\p{N}`` (numbers),
    which Python's ``re`` lacks; ``[^\\W\\d_]`` and ``\\d`` stand in for them
    (they differ only on rare characters such as superscript digits)."""

    PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

    def __init__(self, directory: Path) -> None:
        self.encoder = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        self.decoder = {index: token for token, index in self.encoder.items()}
        lines = (directory / "merges.txt").read_text(encoding="utf-8").split("\n")[1:]
        self.ranks = {tuple(line.split()): rank for rank, line in enumerate(lines) if line}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {character: byte for byte, character in self.byte_encoder.items()}
        self.cache: dict[str, list[str]] = {}

    def _bpe(self, word: str) -> list[str]:
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get(pair, math.inf), i) for i, pair in enumerate(zip(parts, parts[1:]))]
            rank, at = min(pairs)                      # the earliest-learned merge present
            if rank == math.inf:
                break
            first, second = parts[at], parts[at + 1]
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == first and parts[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        self.cache[word] = parts
        return parts

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in self.PATTERN.findall(text):
            mapped = "".join(self.byte_encoder[byte] for byte in word.encode("utf-8"))
            ids.extend(self.encoder[part] for part in self._bpe(mapped))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[index] for index in ids)
        return bytes(self.byte_decoder[character] for character in text).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def read_safetensors(path: Path) -> dict[str, tuple[tuple[int, ...], array]]:
    """{name: (shape, float32 values)}. The format is 8 bytes of header
    length (little-endian), a JSON header giving each tensor's dtype, shape
    and byte range, then the bytes."""

    with open(path, "rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(size))
        blob = memoryview(file.read())
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"{name}: only float32 weights are supported, not {info['dtype']}")
        start, end = info["data_offsets"]
        values = array("f")
        values.frombytes(blob[start:end])
        tensors[name] = (tuple(info["shape"]), values)
    return tensors


def transposed_rows(shape: tuple[int, int], values: array) -> list[array]:
    """The rows of a (rows, columns) matrix's transpose: its columns.

    GPT-2's layers are "Conv1D"s that store (inputs, outputs); TensorLinear
    stores (outputs, inputs)."""

    columns = shape[1]
    return [values[column::columns] for column in range(columns)]


def joined(rows: list[array]) -> array:
    out = array("f")
    for row in rows:
        out.extend(row)
    return out


def read_config(directory: Path) -> dict:
    """Sizes from Hugging Face's config.json (GPT-2 small when there is none):
    the same code then loads gpt2-medium, -large and -xl."""

    path = directory / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    return {"vocab": config.get("vocab_size", 50257), "context": config.get("n_positions", 1024),
            "width": config.get("n_embd", 768), "heads": config.get("n_head", 12),
            "layers": config.get("n_layer", 12)}


def build_model(config: dict, *, context_size: int | None = None, dropout: float = 0.0) -> CharTransformerModel:
    return CharTransformerModel(config["vocab"], context_size=context_size or config["context"],
                                embedding_dim=config["width"], num_heads=config["heads"],
                                num_layers=config["layers"], feed_forward_activation="gelu_tanh",
                                tie_embeddings=True, dropout=dropout)


def load_gpt2(directory: Path, *, context_size: int | None = None, dropout: float = 0.0) -> CharTransformerModel:
    """GPT-2 with OpenAI's weights (``context_size`` keeps the first positions only)."""

    config = read_config(directory)
    # OpenAI's file names tensors "wte.weight"; transformers' save_pretrained, "transformer.wte.weight".
    weights = {name.removeprefix("transformer."): tensor
               for name, tensor in read_safetensors(directory / "model.safetensors").items()}
    model = build_model(config, context_size=context_size, dropout=dropout)
    width = config["width"]

    def plain(name: str) -> array:
        return weights[name][1]

    def linear(layer, name: str) -> None:
        layer.weight.data = joined(transposed_rows(*weights[name + ".weight"]))
        layer.bias.data = plain(name + ".bias")

    def norm(layer, name: str) -> None:
        layer.weight.data = plain(name + ".weight")
        layer.bias.data = plain(name + ".bias")

    model.embedding.token_embedding.weight.data = plain("wte.weight")   # also the output layer (tied)
    model.embedding.position_embedding.weight.data = plain("wpe.weight")[:model.context_size * width]
    model.projection.bias.data = array("f", bytes(4 * config["vocab"]))  # GPT-2's output layer has no bias
    norm(model.norm, "ln_f")
    head = width // config["heads"]
    for index, block in enumerate(model.blocks):
        prefix = f"h.{index}."
        norm(block.norm1, prefix + "ln_1")
        norm(block.norm2, prefix + "ln_2")
        # c_attn holds every head's query, key and value layer side by side:
        # its outputs are [q_0 .. q_11 | k_0 .. k_11 | v_0 .. v_11], 64 each (in GPT-2 small).
        rows = transposed_rows(*weights[prefix + "attn.c_attn.weight"])
        bias = plain(prefix + "attn.c_attn.bias")
        attention = block.attention
        for part, layer in enumerate([*attention.queries, *attention.keys, *attention.values]):
            layer.weight.data = joined(rows[part * head:(part + 1) * head])
            layer.bias.data = bias[part * head:(part + 1) * head]
        linear(attention.projection, prefix + "attn.c_proj")
        linear(block.feed_forward_in, prefix + "mlp.c_fc")
        linear(block.feed_forward_out, prefix + "mlp.c_proj")
    return model


# ---------------------------------------------------------------------------
# Checks against PyTorch
# ---------------------------------------------------------------------------

SAMPLE_TEXT = ("Hello, I'm a language model, and the year is 2019. GPT-2 was trained on 40GB of text "
               "from 8 million web pages; it predicts the next token, one at a time.")


def compare(directory: Path, text: str = SAMPLE_TEXT) -> dict:
    """Tokens, logits, loss and gradients of NanoGrad's GPT-2 and Hugging Face's, on one text.

    Both run in NanoGrad's current precision (float64 on the C backend
    unless NANOGRAD_DTYPE=float32)."""

    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    double = kernels.default_dtype() == "float64"
    reference_tokenizer = AutoTokenizer.from_pretrained(str(directory))
    tokenizer = GPT2Tokenizer(directory)
    ids = tokenizer.encode(text)
    result = {"tokens": len(ids), "same_tokens": ids == reference_tokenizer.encode(text),
              "round_trip": tokenizer.decode(ids) == text}

    reference = GPT2LMHeadModel.from_pretrained(str(directory), attn_implementation="eager")
    reference = reference.double() if double else reference
    reference.eval()                                                   # no dropout
    inputs = torch.tensor([ids[:-1]])
    logits_reference = reference(inputs).logits[0]
    loss_reference = torch.nn.functional.cross_entropy(logits_reference, torch.tensor(ids[1:]))
    loss_reference.backward()

    model = load_gpt2(directory)
    with evaluation_mode(model):
        logits = model.sequence_logits([ids[:-1]])
        loss = sequence_loss(model, [ids[:-1]], [ids[1:]])
        loss.backward()

    ours = list(logits.data)
    theirs = logits_reference.detach().reshape(-1).tolist()
    result["logits_max_difference"] = max(abs(a - b) for a, b in zip(ours, theirs))
    result["logits_largest"] = max(abs(b) for b in theirs)
    result["loss"], result["loss_reference"] = loss.item(), loss_reference.item()

    def gradient_difference(ours_grad, theirs_grad) -> float:
        theirs_list = theirs_grad.reshape(-1).tolist()
        largest = max(abs(v) for v in theirs_list)
        return max(abs(a - b) for a, b in zip(ours_grad, theirs_list)) / largest

    block, hf_block = model.blocks[0], reference.transformer.h[0]
    query = block.attention.queries[0]
    result["grad_relative_difference"] = {
        "token embedding (+ output)": gradient_difference(model.embedding.token_embedding.weight.grad,
                                                          reference.transformer.wte.weight.grad),
        "block 0 query head 0": gradient_difference(query.weight.grad,
                                                    hf_block.attn.c_attn.weight.grad[:, :query.weight.shape[0]].T),
        "block 11 feed-forward out": gradient_difference(model.blocks[-1].feed_forward_out.weight.grad,
                                                         reference.transformer.h[-1].mlp.c_proj.weight.grad.T),
        "final norm": gradient_difference(model.norm.weight.grad, reference.transformer.ln_f.weight.grad),
    }
    return result


# ---------------------------------------------------------------------------
# Sampling and fine-tuning
# ---------------------------------------------------------------------------


def sample(model, tokenizer: GPT2Tokenizer, prompt: str, *, length: int, temperature: float, top_k: int,
           seed: int) -> str:
    ids = generate_tokens(model, tokenizer.encode(prompt), length=length, context_size=model.context_size,
                          temperature=temperature, top_k=top_k, rng=random.Random(seed))
    return tokenizer.decode(ids)


def windows(ids: list[int], context: int, count: int, rng: random.Random) -> tuple[list, list]:
    starts = [rng.randrange(len(ids) - context - 1) for _ in range(count)]
    return [ids[s:s + context] for s in starts], [ids[s + 1:s + context + 1] for s in starts]


def evaluate(model, ids: list[int], context: int, batches: int, batch_size: int, seed: int) -> float:
    rng, total = random.Random(seed), 0.0
    with no_grad(), evaluation_mode(model):
        for _ in range(batches):
            xs, ys = windows(ids, context, batch_size, rng)
            total += sequence_loss(model, xs, ys).item()
    return total / batches


def finetune(args, directory: Path) -> dict:
    """nanoGPT's finetune_shakespeare recipe, for GPT-2 small: a constant small
    learning rate and a few steps of large batches (gradient accumulation)."""

    tokenizer = GPT2Tokenizer(directory)
    text = Path(args.finetune).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"{len(train_ids):,} training and {len(val_ids):,} validation tokens", flush=True)
    model = load_gpt2(directory, context_size=args.context, dropout=args.dropout)
    # 12 layers of (12 x 1,024 x 1,024) attention weights don't fit in 8 GB next to
    # Adam's state: each block recomputes its activations during backward instead.
    model.recompute_blocks = args.recompute
    optimizer = TensorAdam(model.parameters(), lr=args.lr, beta2=0.99, max_grad_norm=1.0)
    rng = random.Random(args.seed)
    measure = lambda: evaluate(model, val_ids, args.context, args.eval_batches, args.batch_size, args.seed + 1)
    rows = [{"step": 0, "val": measure()}]
    print(f"step 0: val loss {rows[0]['val']:.4f}", flush=True)
    start = perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        total = 0.0
        for _ in range(args.grad_accum):
            xs, ys = windows(train_ids, args.context, args.batch_size, rng)
            loss = sequence_loss(model, xs, ys) * (1 / args.grad_accum)
            loss.backward()
            total += loss.item()
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            rows.append({"step": step, "train": total, "val": measure(), "minutes": (perf_counter() - start) / 60})
            print(f"step {step}: train loss {total:.4f}, val loss {rows[-1]['val']:.4f}, "
                  f"{rows[-1]['minutes']:.1f} min", flush=True)
    model.eval()
    print(sample(model, tokenizer, args.prompt or "\n", length=args.length, temperature=args.temperature,
                 top_k=args.top_k, seed=args.seed), flush=True)
    return {"rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default="data/gpt2")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--compare", action="store_true", help="check against Hugging Face's GPT-2")
    parser.add_argument("--sample", metavar="PROMPT", help="continue a prompt")
    parser.add_argument("--finetune", metavar="TEXT_FILE")
    parser.add_argument("--prompt", default=None, help="with --finetune: what to sample after")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=int, default=1024, help="with --finetune: tokens per sequence")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-recompute", dest="recompute", action="store_false",
                        help="with --finetune: keep every block's activations (needs more memory)")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=20)
    args = parser.parse_args(argv)
    directory = Path(args.dir)
    if args.download:
        download(directory)
    if args.compare:
        print(json.dumps(compare(directory), indent=2), flush=True)
    if args.sample is not None:
        model = load_gpt2(directory)
        model.eval()
        start = perf_counter()
        print(sample(model, GPT2Tokenizer(directory), args.sample, length=args.length,
                     temperature=args.temperature, top_k=args.top_k, seed=args.seed))
        print(f"({args.length} tokens in {perf_counter() - start:.1f} s)", flush=True)
    if args.finetune:
        print(json.dumps(finetune(args, directory)["rows"]), flush=True)


if __name__ == "__main__":
    main()
