"""Train NanoGrad and the reference nanoGPT on the same capped text slice.

Both models get the same architecture, the same batches, the same AdamW
settings, and the same top-k sampler with the same random stream, so their
loss curves and sampled text can be compared side by side.

Two initialisation modes:

- ``--init shared`` (default): nanoGPT starts from NanoGrad's exact weights,
  computed in float64. Any gap between the curves is then a real difference
  in the math, not luck, so this mode doubles as an end-to-end correctness
  check of NanoGrad's transformer, autograd, and optimizer.
- ``--init independent``: nanoGPT uses its own init (normal(0, 0.02)) under
  ``torch.manual_seed``. This is the "fair race" comparison.

Requires PyTorch for the nanoGPT side (NanoGrad itself has no dependencies):

    python3 -m venv .venv && .venv/bin/pip install torch
    .venv/bin/python nanogpt_compare.py --text-file data/tinyshakespeare/input.txt

Known, deliberate architecture differences are listed in
docs/language_nanogpt_comparison.md. The one that matters here: NanoGrad's
tied language-model head has an output bias, so this script gives nanoGPT's
``lm_head`` a bias too.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import random
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from char_demo import DEFAULT_TEXT, PRESETS
from language import CharTransformerModel
from losses import softmax_cross_entropy
from optim import TensorAdam
from sampling import sample_from_logits
from tensor import Tensor, no_grad
from text import CharVocab, distinct_ngram_ratio


NANOGPT_MODEL_FILE = Path(__file__).resolve().parent / "ref_nanogpt" / "model.py"
COMPARE_PRESETS = ("tiny-gpt", "lite-gpt", "small-gpt")
FRAMEWORKS = ("nanograd", "nanogpt")


@dataclass(frozen=True)
class CompareConfig:
    """Everything both frameworks must agree on."""

    context_size: int
    embedding_dim: int
    hidden_dim: int
    heads: int
    layers: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    max_grad_norm: float | None
    batch_size: int
    steps: int
    seed: int
    nanograd_init: str = "xavier"


# ---------------------------------------------------------------------------
# Shared data: one text slice, one list of batches
# ---------------------------------------------------------------------------


def split_text(text: str, validation_chars: int, context_size: int) -> tuple[str, str]:
    """Hold out the last ``validation_chars`` characters for validation."""

    if validation_chars <= context_size:
        raise ValueError("validation_chars must be larger than context_size")
    train_text = text[:-validation_chars]
    validation_text = text[-validation_chars:]
    if len(train_text) <= context_size:
        raise ValueError("training text must be longer than context_size")
    return train_text, validation_text


def windows(encoded: list[int], context_size: int) -> list[tuple[list[int], list[int]]]:
    """Every (context, next-character targets) pair, nanoGPT style.

    For ``context_size = 3`` and text "hello": ("hel" -> "ell"), ("ell" -> "llo").
    """

    return [
        (
            encoded[start : start + context_size],
            encoded[start + 1 : start + context_size + 1],
        )
        for start in range(len(encoded) - context_size)
    ]


def training_batches(
    pairs: list[tuple[list[int], list[int]]],
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[tuple[list[list[int]], list[list[int]]]]:
    """Random batches of windows, drawn once so both frameworks share them."""

    rng = random.Random(seed)
    batches = []
    for _ in range(steps):
        chosen = [rng.randrange(len(pairs)) for _ in range(batch_size)]
        batches.append((
            [pairs[index][0] for index in chosen],
            [pairs[index][1] for index in chosen],
        ))
    return batches


# ---------------------------------------------------------------------------
# NanoGrad side
# ---------------------------------------------------------------------------


def build_nanograd_model(config: CompareConfig, vocab_size: int) -> CharTransformerModel:
    return CharTransformerModel(
        vocab_size,
        context_size=config.context_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.heads,
        num_layers=config.layers,
        feed_forward_activation="gelu",
        tie_embeddings=True,
        init=config.nanograd_init,
        seed=config.seed,
    )


class NanoGradRunner:
    """Train / evaluate / sample a NanoGrad character transformer."""

    def __init__(self, config: CompareConfig, vocab_size: int) -> None:
        self.model = build_nanograd_model(config, vocab_size)
        self.optimizer = TensorAdam(
            self.model.parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            weight_decay_min_ndim=2,
            max_grad_norm=config.max_grad_norm,
        )

    def train_step(self, xs: list[list[int]], ys: list[list[int]]) -> float:
        loss = self._loss(xs, ys)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, xs: list[list[int]], ys: list[list[int]]) -> float:
        with no_grad():
            return self._loss(xs, ys).item()

    def next_logits(self, context: list[int]) -> list[float]:
        with no_grad():
            return self.model(Tensor.from_list([context])).data

    def _loss(self, xs: list[list[int]], ys: list[list[int]]) -> Tensor:
        logits = self.model.sequence_logits(Tensor.from_list(xs))
        batch, context, vocab = logits.shape
        targets = [target for row in ys for target in row]
        return softmax_cross_entropy(logits.reshape((batch * context, vocab)), targets)


# ---------------------------------------------------------------------------
# nanoGPT side (PyTorch)
# ---------------------------------------------------------------------------


def load_nanogpt_module():
    """Import ref_nanogpt/model.py under a name that cannot clash with model.py."""

    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "nanogpt_compare.py needs PyTorch for the nanoGPT side: pip install torch"
        )

    spec = importlib.util.spec_from_file_location("ref_nanogpt_model", NANOGPT_MODEL_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nanogpt_state_from_nanograd(model: CharTransformerModel) -> dict[str, list]:
    """Map NanoGrad parameters onto nanoGPT parameter names as nested lists.

    The only real reshuffle is attention: nanoGPT fuses every head's query,
    key, and value projection into one ``c_attn`` matrix whose rows are

        [ q_head0 ; q_head1 ; ... ; k_head0 ; ... ; v_head0 ; ... ]

    NanoGrad keeps one small (head_dim, embedding_dim) layer per head.
    Linear weights use the same (out, in) layout in both frameworks.
    """

    state: dict[str, list] = {
        "transformer.wte.weight": model.embedding.token_embedding.weight.tolist(),
        "transformer.wpe.weight": model.embedding.position_embedding.weight.tolist(),
        "transformer.ln_f.weight": model.norm.weight.tolist(),
        "transformer.ln_f.bias": model.norm.bias.tolist(),
        "lm_head.bias": model.projection.bias.tolist(),
    }
    for index, block in enumerate(model.blocks):
        prefix = f"transformer.h.{index}."
        attention = block.attention
        fused_layers = [*attention.queries, *attention.keys, *attention.values]
        state[prefix + "ln_1.weight"] = block.norm1.weight.tolist()
        state[prefix + "ln_1.bias"] = block.norm1.bias.tolist()
        state[prefix + "attn.c_attn.weight"] = [
            row
            for layer in fused_layers
            for row in layer.weight.tolist()
        ]
        state[prefix + "attn.c_attn.bias"] = [
            value
            for layer in fused_layers
            for value in layer.bias.tolist()
        ]
        state[prefix + "attn.c_proj.weight"] = attention.projection.weight.tolist()
        state[prefix + "attn.c_proj.bias"] = attention.projection.bias.tolist()
        state[prefix + "ln_2.weight"] = block.norm2.weight.tolist()
        state[prefix + "ln_2.bias"] = block.norm2.bias.tolist()
        state[prefix + "mlp.c_fc.weight"] = block.feed_forward_in.weight.tolist()
        state[prefix + "mlp.c_fc.bias"] = block.feed_forward_in.bias.tolist()
        state[prefix + "mlp.c_proj.weight"] = block.feed_forward_out.weight.tolist()
        state[prefix + "mlp.c_proj.bias"] = block.feed_forward_out.bias.tolist()
    return state


class NanoGPTRunner:
    """Train / evaluate / sample the reference nanoGPT ``GPT`` in float64."""

    def __init__(
        self,
        config: CompareConfig,
        vocab_size: int,
        *,
        init_from: CharTransformerModel | None,
    ) -> None:
        import torch
        from torch import nn

        nanogpt = load_nanogpt_module()
        torch.set_default_dtype(torch.float64)
        # Tiny models gain nothing from a thread pool, and its busy-waiting
        # threads would steal CPU from NanoGrad's single Python thread.
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        if config.hidden_dim != 4 * config.embedding_dim:
            raise ValueError("nanoGPT's MLP is fixed at 4 * embedding_dim")

        gpt_config = nanogpt.GPTConfig(
            block_size=config.context_size,
            vocab_size=vocab_size,
            n_layer=config.layers,
            n_head=config.heads,
            n_embd=config.embedding_dim,
            dropout=0.0,
            bias=True,
        )
        with redirect_stdout(io.StringIO()):  # GPT() prints its parameter count
            self.model = nanogpt.GPT(gpt_config)

        # Match NanoGrad: a tied lm_head that also has an output bias.
        head = nn.Linear(config.embedding_dim, vocab_size, bias=True)
        nn.init.zeros_(head.bias)
        head.weight = self.model.transformer.wte.weight
        self.model.lm_head = head

        if init_from is not None:
            state = nanogpt_state_from_nanograd(init_from)
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    parameter.copy_(torch.tensor(state[name]))

        self.torch = torch
        self.max_grad_norm = config.max_grad_norm
        self.parameters = list(self.model.parameters())
        # nanoGPT's configure_optimizers rule: decay matrices, not vectors.
        groups = [
            {
                "params": [p for p in self.parameters if p.dim() >= 2],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for p in self.parameters if p.dim() < 2],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
        )

    def train_step(self, xs: list[list[int]], ys: list[list[int]]) -> float:
        torch = self.torch
        self.model.train()
        _, loss = self.model(torch.tensor(xs), torch.tensor(ys))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, xs: list[list[int]], ys: list[list[int]]) -> float:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            _, loss = self.model(torch.tensor(xs), torch.tensor(ys))
        return loss.item()

    def next_logits(self, context: list[int]) -> list[float]:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(torch.tensor([context]))
        return logits[0, -1].tolist()

    def _clip_gradients(self) -> None:
        """Global-norm clipping written exactly like NanoGrad's optimizer.

        (torch's clip_grad_norm_ divides by norm + 1e-6, a tiny difference
        that would otherwise show up in the shared-init curves.)
        """

        if self.max_grad_norm is None:
            return
        grads = [p.grad for p in self.parameters if p.grad is not None]
        norm = math.sqrt(sum(float((grad * grad).sum()) for grad in grads))
        if norm > self.max_grad_norm:
            for grad in grads:
                grad.mul_(self.max_grad_norm / norm)


# ---------------------------------------------------------------------------
# Sampling with one shared sampler
# ---------------------------------------------------------------------------


def generate(
    next_logits,
    vocab: CharVocab,
    *,
    seed_text: str,
    context_size: int,
    length: int,
    temperature: float,
    top_k: int | None,
    rng: random.Random,
) -> str:
    """Autoregressive sampling over a fixed-size sliding context window."""

    text = seed_text
    for _ in range(length):
        context = vocab.encode(text[-context_size:])
        logits = next_logits(context)
        index = sample_from_logits(
            logits[: len(vocab)],
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        text += vocab.decode([index])
    return text


def matching_prefix_length(left: str, right: str) -> int:
    """How many leading characters two strings share."""

    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    preset = PRESETS[args.preset]
    config = CompareConfig(
        context_size=preset["context_size"],
        embedding_dim=preset["embedding_dim"],
        hidden_dim=4 * preset["embedding_dim"],
        heads=preset["heads"],
        layers=preset["layers"],
        lr=args.lr if args.lr is not None else preset["lr"],
        weight_decay=preset["weight_decay"],
        beta1=args.beta1,
        beta2=args.beta2,
        max_grad_norm=preset["max_grad_norm"],
        batch_size=args.batch_size or preset["batch_size"],
        steps=args.steps,
        seed=args.seed,
        nanograd_init=args.nanograd_init,
    )

    text = _load_text(args)
    train_text, validation_text = split_text(
        text,
        args.validation_chars,
        config.context_size,
    )
    vocab = CharVocab.from_text(text)
    train_pairs = windows(vocab.encode(train_text), config.context_size)
    validation_pairs = windows(vocab.encode(validation_text), config.context_size)
    batches = training_batches(
        train_pairs,
        batch_size=config.batch_size,
        steps=config.steps,
        seed=config.seed,
    )
    validation_xs = [pair[0] for pair in validation_pairs]
    validation_ys = [pair[1] for pair in validation_pairs]

    nanograd = NanoGradRunner(config, len(vocab))
    nanogpt = NanoGPTRunner(
        config,
        len(vocab),
        init_from=nanograd.model if args.init == "shared" else None,
    )
    runners = {"nanograd": nanograd, "nanogpt": nanogpt}

    metrics: list[dict] = []
    seconds = {name: 0.0 for name in FRAMEWORKS}
    for step, (xs, ys) in enumerate(batches, start=1):
        for name, runner in runners.items():
            start = perf_counter()
            train_loss = runner.train_step(xs, ys)
            seconds[name] += perf_counter() - start
            if step % args.eval_every == 0 or step == config.steps:
                metrics.append({
                    "step": step,
                    "framework": name,
                    "train_loss": train_loss,
                    "validation_loss": runner.evaluate(validation_xs, validation_ys),
                })
        if step % args.eval_every == 0 or step == config.steps:
            _print_step(metrics[-2:])

    seed_text = args.seed_text or train_text[: config.context_size]
    if len(seed_text) < config.context_size:
        raise ValueError("seed_text must be at least context_size characters")

    samples: list[dict] = []
    for sample_index in range(args.num_samples):
        for name, runner in runners.items():
            text_out = generate(
                runner.next_logits,
                vocab,
                seed_text=seed_text,
                context_size=config.context_size,
                length=args.generate,
                temperature=args.temperature,
                top_k=args.top_k,
                # Same stream for both frameworks: equal logits -> equal text.
                rng=random.Random(args.sample_seed + sample_index),
            )
            samples.append({
                "framework": name,
                "sample_index": sample_index,
                "seed_prefix": seed_text,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "text": text_out,
                "distinct_2": distinct_ngram_ratio(text_out, n=2),
            })

    summary = _summaries(metrics, samples, seconds, config)
    _print_report(args, config, summary, samples)
    if args.output_dir is not None:
        write_outputs(args, config, metrics, samples, summary)
    return {"metrics": metrics, "samples": samples, "summary": summary}


def _summaries(
    metrics: list[dict],
    samples: list[dict],
    seconds: dict[str, float],
    config: CompareConfig,
) -> list[dict]:
    rows = []
    for name in FRAMEWORKS:
        final = [row for row in metrics if row["framework"] == name][-1]
        texts = [row["text"] for row in samples if row["framework"] == name]
        rows.append({
            "framework": name,
            "steps": config.steps,
            "final_train_loss": final["train_loss"],
            "final_validation_loss": final["validation_loss"],
            "validation_perplexity": math.exp(final["validation_loss"]),
            "mean_distinct_2": sum(distinct_ngram_ratio(t) for t in texts) / len(texts),
            "train_seconds": seconds[name],
            "tokens_per_second": (
                config.steps * config.batch_size * config.context_size / seconds[name]
            ),
        })

    nanograd_texts = [row["text"] for row in samples if row["framework"] == "nanograd"]
    nanogpt_texts = [row["text"] for row in samples if row["framework"] == "nanogpt"]
    shared = [
        matching_prefix_length(a, b) / len(a)
        for a, b in zip(nanograd_texts, nanogpt_texts)
    ]
    by_step: dict[int, dict[str, float]] = {}
    for row in metrics:
        by_step.setdefault(row["step"], {})[row["framework"]] = row["validation_loss"]
    loss_gap = max(abs(pair["nanograd"] - pair["nanogpt"]) for pair in by_step.values())
    for row in rows:
        row["mean_shared_sample_prefix"] = sum(shared) / len(shared)
        row["max_validation_loss_gap"] = loss_gap
    return rows


def write_outputs(
    args: argparse.Namespace,
    config: CompareConfig,
    metrics: list[dict],
    samples: list[dict],
    summary: list[dict],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", metrics)
    _write_csv(output_dir / "samples.csv", samples)
    _write_csv(output_dir / "summary.csv", summary)
    manifest = {
        "text_source": str(args.text_file) if args.text_file else "built-in",
        "max_chars": args.max_chars,
        "validation_chars": args.validation_chars,
        "preset": args.preset,
        "init": args.init,
        "config": asdict(config),
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "num_samples": args.num_samples,
            "generate": args.generate,
            "sample_seed": args.sample_seed,
        },
        "outputs": ["metrics.csv", "samples.csv", "summary.csv"],
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_step(rows: list[dict]) -> None:
    parts = [
        f"{row['framework']}: train {row['train_loss']:.4f} val {row['validation_loss']:.4f}"
        for row in rows
    ]
    print(f"step {rows[0]['step']:>5}  " + "  |  ".join(parts))


def _print_report(
    args: argparse.Namespace,
    config: CompareConfig,
    summary: list[dict],
    samples: list[dict],
) -> None:
    print()
    print(f"preset {args.preset}, init {args.init}, {config.steps} steps")
    print(f"{'framework':<10} {'val loss':>9} {'val ppl':>8} {'dist-2':>7} {'tok/s':>9}")
    for row in summary:
        print(
            f"{row['framework']:<10} {row['final_validation_loss']:>9.4f} "
            f"{row['validation_perplexity']:>8.3f} {row['mean_distinct_2']:>7.3f} "
            f"{row['tokens_per_second']:>9.1f}"
        )
    print(f"max val-loss gap:     {summary[0]['max_validation_loss_gap']:.2e}")
    print(f"shared sample prefix: {summary[0]['mean_shared_sample_prefix']:.1%}")
    print()
    for row in samples:
        print(f"[{row['framework']} #{row['sample_index']}] {row['text']!r}")


def _load_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = DEFAULT_TEXT * 8
    return text[: args.max_chars]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--max-chars", type=int, default=2048)
    parser.add_argument("--validation-chars", type=int, default=256)
    parser.add_argument("--preset", choices=COMPARE_PRESETS, default="small-gpt")
    parser.add_argument("--init", choices=("shared", "independent"), default="shared")
    parser.add_argument(
        "--nanograd-init",
        choices=("xavier", "gpt2"),
        default="xavier",
        help="NanoGrad's initializer (with --init shared, nanoGPT copies it)",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-text")
    parser.add_argument("--generate", type=int, default=120)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.eval_every <= 0 or args.num_samples <= 0:
        parser.error("--steps, --eval-every, and --num-samples must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
