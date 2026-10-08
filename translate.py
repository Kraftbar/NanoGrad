"""German to English translation on Multi30k with the encoder-decoder transformer.

Multi30k (Elliott et al., 2016) pairs 31,014 image descriptions in German
with English translations (29,000 to train on, 1,014 to validate, the
1,000-sentence "test 2016" set to test). It is the usual first
translation benchmark, small enough for a CPU.

Translations are scored with BLEU (Papineni et al., 2002), the standard
machine-translation metric: the geometric mean of how many of the
output's 1, 2, 3 and 4-word sequences also appear in the reference
translation, times a penalty for outputs shorter than the references,
over the whole test set, as a percentage.

    python3 translate.py --epochs 10
    .venv/bin/python translate.py --compare --steps 20    # same weights in PyTorch

Data: data/multi30k/ (see data/multi30k/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

from losses import softmax_cross_entropy
from optim import TensorAdam
from schedules import warmup_cosine
from seq2seq import BOS, EOS, PAD, Seq2SeqTransformer, greedy_translate, pad
from tensor import Tensor, gather_rows, no_grad
from train import evaluation_mode


TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(sentence: str) -> list[str]:
    return TOKEN.findall(sentence.lower())


class Vocab:
    """Words seen at least ``min_count`` times; others become <unk>."""

    SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, sentences: list[list[str]], *, min_count: int = 2) -> None:
        counts = Counter(word for sentence in sentences for word in sentence)
        self.words = list(self.SPECIALS) + sorted(word for word, count in counts.items() if count >= min_count)
        self.ids = {word: index for index, word in enumerate(self.words)}

    def encode(self, sentence: list[str]) -> list[int]:
        return [self.ids.get(word, 3) for word in sentence]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.words[index] for index in ids if index not in (PAD, BOS, EOS)]

    def __len__(self) -> int:
        return len(self.words)


def load_pairs(folder: Path, split: str) -> list[tuple[list[str], list[str]]]:
    german = (folder / f"{split}.de").read_text(encoding="utf-8").splitlines()
    english = (folder / f"{split}.en").read_text(encoding="utf-8").splitlines()
    return [(tokenize(de), tokenize(en)) for de, en in zip(german, english)]


def batches(pairs: list[tuple[list[int], list[int]]], batch_size: int, rng: random.Random) -> list[list]:
    """Batches of sentences of similar length (less padding), in random order."""

    order = sorted(range(len(pairs)), key=lambda index: (len(pairs[index][0]), rng.random()))
    groups = [[pairs[index] for index in order[start : start + batch_size]]
              for start in range(0, len(order), batch_size)]
    rng.shuffle(groups)
    return groups


def batch_loss(model: Seq2SeqTransformer, batch, *, label_smoothing: float = 0.0) -> Tensor:
    """Mean next-token loss over the real (non-padding) target positions.

    The decoder reads <bos> y1 .. yn and must predict y1 .. yn <eos>."""

    source = pad([src + [EOS] for src, _ in batch])
    target_in = pad([[BOS] + tgt for _, tgt in batch])
    target_out = pad([tgt + [EOS] for _, tgt in batch])
    scores = model(source, target_in)                                     # (B, T, V)
    batch_size, length, vocab = scores.shape
    positions = [row * length + step for row, out in enumerate(target_out) for step, token in enumerate(out)
                 if token != PAD]
    labels = [target_out[position // length][position % length] for position in positions]
    picked = gather_rows(scores.reshape((batch_size * length, vocab)), positions)
    return softmax_cross_entropy(picked, labels, label_smoothing=label_smoothing)


def bleu(hypotheses: list[list[str]], references: list[list[str]], max_n: int = 4) -> float:
    """Corpus BLEU-4 in percent: clipped n-gram precisions, geometric mean,
    brevity penalty (one reference per sentence)."""

    matches, totals = [0] * max_n, [0] * max_n
    hypothesis_length = reference_length = 0
    for hypothesis, reference in zip(hypotheses, references):
        hypothesis_length += len(hypothesis)
        reference_length += len(reference)
        for n in range(1, max_n + 1):
            grams = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
            reference_grams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
            matches[n - 1] += sum(min(count, reference_grams[gram]) for gram, count in grams.items())
            totals[n - 1] += max(0, len(hypothesis) - n + 1)
    if min(matches) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_n
    brevity = min(0.0, 1.0 - reference_length / hypothesis_length)
    return 100.0 * math.exp(log_precision + brevity)


def evaluate(model, pairs, source_vocab: Vocab, target_vocab: Vocab, *, limit: int | None = None) -> float:
    pairs = pairs[:limit]
    hypotheses = []
    for start in range(0, len(pairs), 100):
        chunk = pairs[start : start + 100]
        sources = [source_vocab.encode(src) + [EOS] for src, _ in chunk]
        outputs = greedy_translate(model, sources, max_length=max(len(src) for src in sources) + 10)
        hypotheses += [target_vocab.decode(out) for out in outputs]
    return bleu(hypotheses, [tgt for _, tgt in pairs])


def build(args, source_vocab: Vocab, target_vocab: Vocab, *, dropout: float | None = None) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(
        len(source_vocab), len(target_vocab), context_size=args.context, width=args.width, heads=args.heads,
        hidden=args.hidden, encoder_layers=args.layers, decoder_layers=args.layers,
        dropout=args.dropout if dropout is None else dropout, seed=args.seed,
    )


def run(args) -> dict:
    folder = Path(args.data)
    train_words, valid_words, test_words = (load_pairs(folder, split) for split in ("train", "val", "test_2016_flickr"))
    source_vocab = Vocab([src for src, _ in train_words])
    target_vocab = Vocab([tgt for _, tgt in train_words])
    keep = [(src, tgt) for src, tgt in train_words if len(src) < args.context and len(tgt) < args.context]
    train_set = [(source_vocab.encode(src), target_vocab.encode(tgt)) for src, tgt in keep][: args.train_limit]
    print(f"{len(train_set)} training pairs; German vocabulary {len(source_vocab)}, English {len(target_vocab)}",
          flush=True)

    model = build(args, source_vocab, target_vocab)
    print(f"model {model.num_parameters()} parameters", flush=True)
    optimizer = TensorAdam(model.parameters(), lr=args.lr, beta2=0.98, eps=1e-9, max_grad_norm=1.0)
    rng = random.Random(args.seed)
    steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
    schedule = warmup_cosine(max_lr=args.lr, min_lr=args.lr / 10, warmup_steps=args.warmup,
                             decay_steps=args.epochs * steps_per_epoch)
    step, history = 0, []
    for epoch in range(1, args.epochs + 1):
        start, losses = perf_counter(), []
        model.train()
        for batch in batches(train_set, args.batch_size, rng):
            optimizer.lr = schedule(step)
            optimizer.zero_grad()
            loss = batch_loss(model, batch, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
        with no_grad(), evaluation_mode(model):
            valid_loss = sum(batch_loss(model, batch).item() for batch in batches(
                [(source_vocab.encode(s), target_vocab.encode(t)) for s, t in valid_words], 100, random.Random(0)))
        valid_loss /= math.ceil(len(valid_words) / 100)
        score = evaluate(model, valid_words, source_vocab, target_vocab, limit=args.bleu_limit)
        minutes = (perf_counter() - start) / 60
        print(f"epoch {epoch:2d}: train loss {sum(losses) / len(losses):.3f}, valid loss {valid_loss:.3f}, "
              f"valid BLEU {score:.1f}, {minutes:.1f} min", flush=True)
        history.append({"epoch": epoch, "valid_bleu": score, "valid_loss": valid_loss})
    test = evaluate(model, test_words, source_vocab, target_vocab)
    print(f"test 2016 BLEU {test:.1f}", flush=True)
    for src, tgt in test_words[:3]:
        out = greedy_translate(model, [source_vocab.encode(src) + [EOS]], max_length=len(src) + 10)[0]
        print(f"  {' '.join(src)}\n  -> {' '.join(target_vocab.decode(out))}\n     ({' '.join(tgt)})")
    return {"history": history, "test_bleu": test}


def compare(args) -> dict:
    """NanoGrad and PyTorch (torch_twin.py) from the same weights on the same
    batches, without dropout: losses, time per step, and weight drift."""

    import torch

    from torch_twin import TorchTwin

    torch.set_num_threads(1)
    folder = Path(args.data)
    train_words = load_pairs(folder, "train")
    source_vocab, target_vocab = Vocab([s for s, _ in train_words]), Vocab([t for _, t in train_words])
    pairs = [(source_vocab.encode(s), target_vocab.encode(t)) for s, t in train_words
             if len(s) < args.context and len(t) < args.context]
    chosen = batches(pairs, args.batch_size, random.Random(args.seed))[: args.steps]
    model = build(args, source_vocab, target_vocab, dropout=0.0)
    twin = TorchTwin(model)                                             # copies the initial weights
    ours = TensorAdam(model.parameters(), lr=args.lr, beta2=0.98, eps=1e-9, max_grad_norm=1.0)
    theirs = torch.optim.Adam(twin.parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    results = {}
    with no_grad():                     # build each shape's index tables and masks before timing
        for batch in chosen:
            batch_loss(model, batch)

    start, losses = perf_counter(), []
    for batch in chosen:
        ours.zero_grad()
        loss = batch_loss(model, batch, label_smoothing=args.label_smoothing)
        loss.backward()
        ours.step()
        losses.append(loss.item())
    results["nanograd"] = {"ms_per_step": 1000 * (perf_counter() - start) / len(chosen), "losses": losses}

    start, losses = perf_counter(), []
    for batch in chosen:
        source = pad([src + [EOS] for src, _ in batch])
        target_in, target_out = pad([[BOS] + t for _, t in batch]), pad([t + [EOS] for _, t in batch])
        theirs.zero_grad()
        scores = twin.seq2seq(model, source, target_in)
        loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), torch.tensor(target_out).reshape(-1),
                                                 ignore_index=PAD, label_smoothing=args.label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(twin.parameters, 1.0)
        theirs.step()
        losses.append(loss.item())
    results["torch"] = {"ms_per_step": 1000 * (perf_counter() - start) / len(chosen), "losses": losses}

    for name, result in results.items():
        print(f"{name:8s}: {result['ms_per_step']:7.1f} ms/step, loss {result['losses'][0]:.4f} -> "
              f"{result['losses'][-1]:.4f}")
    difference = max(abs(a - b) for p, t in zip(model.parameters(), twin.parameters)
                     for a, b in zip(p.data, t.detach().reshape(-1).tolist()))
    print(f"largest weight difference after {len(chosen)} steps: {difference:.1e}")
    results["max_weight_difference"] = difference
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/multi30k")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--bleu-limit", type=int, default=300, help="validation sentences scored each epoch")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--compare", action="store_true", help="NanoGrad vs PyTorch on the same steps")
    parser.add_argument("--steps", type=int, default=20, help="--compare: training steps")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    compare(args) if args.compare else run(args)


if __name__ == "__main__":
    main()
