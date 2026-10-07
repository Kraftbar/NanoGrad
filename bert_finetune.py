"""Pretrain the tiny BERT encoder, then fine-tune it to recognize speakers.

The GLUE-style workflow, end to end, on Tiny Shakespeare (see speakers.py):

1. pretrain: masked character modeling (bert.py) on the text, with the
   held-out speeches removed
2. fine-tune: put a classifier on the encoder and train everything to
   predict which of 8 characters speaks a 64-character window
3. baseline: the same classifier on a fresh, untrained encoder, trained
   for the same number of steps

Accuracy on held-out speeches shows what pretraining bought (chance is 1/8).
With ``--probe`` the encoder is frozen and only the classifier layer
trains (a "linear probe"), which measures how much the pretrained
features alone say about the speaker. With ``--lora 4`` the encoder is
frozen too, but rank-4 LoRA adapters on its attention layers train along
with the classifier (lora.py).

    python3 bert_finetune.py --pretrain-steps 2000 --finetune-steps 1000
    python3 bert_finetune.py --probe
    python3 bert_finetune.py --lora 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter

from bert import BERTClassifier, TinyBERT, mask_tokens, masked_loss
from gpt_train import random_batch
from lora import add_lora
from losses import softmax_cross_entropy
from optim import TensorAdam
from speakers import speaker_data, speech_windows
from tensor import no_grad
from text import CharVocab
from train import evaluation_mode


def build_encoder(vocab_size: int, args: argparse.Namespace) -> TinyBERT:
    return TinyBERT(
        vocab_size, context_size=args.context_size, embedding_dim=args.embedding_dim,
        num_heads=args.heads, num_layers=args.layers, seed=args.seed,
    )


def pretrain(encoder: TinyBERT, encoded: list[int], args: argparse.Namespace) -> None:
    optimizer = TensorAdam(encoder.parameters(), lr=args.lr, weight_decay=0.01, weight_decay_min_ndim=2)
    rng = random.Random(args.seed)
    start = perf_counter()
    recent = []
    for step in range(1, args.pretrain_steps + 1):
        windows, _ = random_batch(encoded, batch_size=args.batch_size, context_size=args.context_size, rng=rng)
        batch = mask_tokens(windows, vocab_size=encoder.vocab_size, mask_id=encoder.mask_id, rng=rng)
        optimizer.zero_grad()
        loss = masked_loss(encoder, *batch)
        loss.backward()
        optimizer.step()
        recent.append(loss.item())
        if step % args.report_every == 0:
            print(f"pretrain step {step:5d}: masked-character loss {sum(recent) / len(recent):.4f}, "
                  f"{perf_counter() - start:.0f} s", flush=True)
            recent = []


def fine_tune(model: BERTClassifier, vocab: CharVocab, data, args: argparse.Namespace, label: str) -> float:
    if args.lora:
        trainable = add_lora(model.encoder, rank=args.lora, seed=args.seed) + model.classifier.parameters()
    else:
        trainable = model.classifier.parameters() if args.probe else model.parameters()
    print(f"{label}: training {sum(parameter.numel for parameter in trainable)} parameters", flush=True)
    optimizer = TensorAdam(trainable, lr=args.finetune_lr, weight_decay=0.01, weight_decay_min_ndim=2)
    rng = random.Random(args.seed + 10)
    start = perf_counter()
    for step in range(1, args.finetune_steps + 1):
        texts, labels = speech_windows(data.train, length=args.context_size, count=args.batch_size, rng=rng)
        optimizer.zero_grad()
        loss = softmax_cross_entropy(scores(model, [vocab.encode(text) for text in texts], args.probe), labels)
        loss.backward()
        optimizer.step()
        if step % args.report_every == 0:
            print(f"{label:12s} step {step:5d}: loss {loss.item():.4f}, "
                  f"{perf_counter() - start:.0f} s", flush=True)
    return accuracy(model, vocab, data, args)


def scores(model: BERTClassifier, batch: list[list[int]], probe: bool):
    """Class scores; a probe computes the frozen encoder's features without gradients."""

    if not probe:
        return model(batch)
    with no_grad():
        features = model.encoder.encode(batch).mean(axis=1)
    return model.classifier(features)


def accuracy(model: BERTClassifier, vocab: CharVocab, data, args: argparse.Namespace) -> float:
    texts, labels = speech_windows(data.validation, length=args.context_size,
                                   count=args.eval_windows, rng=random.Random(args.seed + 20))
    correct = 0
    for start in range(0, len(texts), 100):
        batch = [vocab.encode(text) for text in texts[start : start + 100]]
        with no_grad(), evaluation_mode(model):
            logits = model(batch)
        classes = logits.shape[1]
        for row, expected in enumerate(labels[start : start + 100]):
            scores = logits.data[row * classes : (row + 1) * classes]
            correct += max(range(classes), key=scores.__getitem__) == expected
    return correct / len(texts)


def run(args: argparse.Namespace) -> dict:
    text = Path(args.text_file).read_text(encoding="utf-8")
    vocab = CharVocab.from_text(text)
    data = speaker_data(text, num_speakers=args.speakers)
    print(f"speakers: {', '.join(data.speakers)}")
    print(f"{len(data.train)} training speeches, {len(data.validation)} held out; "
          f"pretraining on {len(data.pretraining_text)} characters")

    pretrained = build_encoder(len(vocab), args)
    pretrain(pretrained, vocab.encode(data.pretraining_text), args)
    results = {
        "pretrained": fine_tune(BERTClassifier(pretrained, len(data.speakers), seed=args.seed), vocab, data, args, "pretrained"),
        "untrained": fine_tune(
            BERTClassifier(build_encoder(len(vocab), args), len(data.speakers), seed=args.seed),
            vocab, data, args, "untrained",
        ),
    }
    for name, value in results.items():
        print(f"{name:12s}: {100 * value:.1f}% of held-out windows (chance {100 / len(data.speakers):.1f}%)")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text-file", default="data/tinyshakespeare/input.txt")
    parser.add_argument("--speakers", type=int, default=8)
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--finetune-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true", help="freeze the encoder, train only the classifier")
    mode.add_argument("--lora", type=int, default=0, metavar="RANK",
                      help="freeze the encoder, train LoRA adapters of this rank and the classifier")
    parser.add_argument("--eval-windows", type=int, default=2000)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
