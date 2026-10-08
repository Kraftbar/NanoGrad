"""Preference tuning with DPO (Rafailov et al., 2023) on the baby GPT.

After pretraining, chat models are tuned on preferences: pairs of answers
to the same prompt where people marked one as better. RLHF fits a reward
model to those pairs and then runs reinforcement learning against it.
Direct Preference Optimization skips both steps. It notes that the
optimal policy under a KL penalty to the starting model ("reference")
gives every answer an implicit reward,

    r(x, y) = beta * (log pi(y | x) - log pi_ref(y | x)),

and fits that reward to the pairs directly, as a classifier of which
answer won:

    loss = -log sigmoid(r(x, y_win) - r(x, y_lose))

It is ordinary supervised training: two forward passes of the model, and
the sums of the log-probabilities of each answer's tokens.

The demonstration needs preferences that can be checked automatically.
The baby GPT (`gpt_train.py --preset baby-llama`, character-level Tiny
Shakespeare) often writes plausible-looking non-words. So the "labeler"
prefers the continuation with more real words: words that occur in
Shakespeare's text count +1, invented ones -1. Two continuations per
prompt are sampled from the model itself, and the one with the higher
score wins. DPO should make the model write more real words without
retraining it on text, while the log-probability ratio to the reference
(the KL) stays modest.

    python3 gpt_train.py --preset baby-llama --out-dir runs/baby_llama     # the starting model
    python3 dpo.py --run runs/baby_llama
    .venv/bin/python dpo.py --run runs/baby_llama --compare               # one DPO step in PyTorch too
"""

from __future__ import annotations

import argparse
import copy
import random
import re
from pathlib import Path
from time import perf_counter

from gpt_train import generate_tokens, load_run, split_text
from losses import binary_cross_entropy_with_logits
from optim import TensorAdam
from tensor import Tensor, gather_rows, no_grad
from text import CharVocab
from train import evaluation_mode

WORD = re.compile(r"[a-z]+")


def word_score(text: str, known: set[str]) -> int:
    """+1 for every complete real word, -1 for every invented one. Only words
    with a non-letter on both sides count (the ends of a continuation may cut
    a word in half)."""

    score = 0
    for match in WORD.finditer(text.lower()):
        if match.start() > 0 and match.end() < len(text):
            score += 1 if match.group() in known else -1
    return score


def sequence_log_probs(model, sequences: list[list[int]], prompt_length: int) -> Tensor:
    """log pi(continuation | prompt) for each sequence: the sum of the
    log-probabilities of its tokens after the prompt. Shape (len(sequences),)."""

    batch, length = len(sequences), len(sequences[0])
    logits = model.sequence_logits(Tensor.from_list([s[:-1] for s in sequences]))     # (B, T - 1, V)
    vocab = logits.shape[-1]
    flat = logits.reshape((batch * (length - 1), vocab))
    # log softmax = x - log sum exp(x), with each row's max subtracted first so nothing overflows or underflows
    rows = batch * (length - 1)
    peaks = Tensor([max(flat.data[r * vocab : (r + 1) * vocab]) for r in range(rows)], (rows, 1))
    shifted = flat - peaks
    log_probs = shifted - shifted.exp().sum(axis=1).log().reshape((rows, 1))
    # Row b, position t predicts token t + 1; keep the continuation's tokens only.
    picks = [(b * (length - 1) + t) * vocab + sequences[b][t + 1]
             for b in range(batch) for t in range(prompt_length - 1, length - 1)]
    chosen = gather_rows(log_probs.reshape((batch * (length - 1) * vocab, 1)), picks)
    return chosen.reshape((batch, length - prompt_length)).sum(axis=1)


def dpo_loss(policy, reference, winners, losers, prompt_length: int, beta: float) -> tuple[Tensor, float]:
    """The DPO loss of a batch of pairs, and the fraction the policy already ranks right."""

    with no_grad(), evaluation_mode(reference):
        ref = list(sequence_log_probs(reference, winners + losers, prompt_length).data)
    pol = sequence_log_probs(policy, winners + losers, prompt_length)
    count = len(winners)
    ref_margin = Tensor([ref[i] - ref[count + i] for i in range(count)], (count,))
    margin = (pol[:count] - pol[count:] - ref_margin) * beta          # r(x, y_win) - r(x, y_lose)
    loss = binary_cross_entropy_with_logits(margin, Tensor([1.0] * count, (count,)))   # -log sigmoid(margin)
    return loss, sum(m > 0 for m in margin.data) / count


def make_pairs(model, vocab, text_tokens: list[int], known: set[str], args, rng: random.Random):
    """(prompt + winner, prompt + loser) token lists, from two samples per prompt."""

    pairs = []
    while len(pairs) < args.pairs:
        start = rng.randrange(len(text_tokens) - args.prompt_length)
        prompt = text_tokens[start : start + args.prompt_length]
        samples = [generate_tokens(model, prompt, length=args.continuation, context_size=model.context_size,
                                   temperature=1.0, top_k=None, rng=rng) for _ in range(2)]
        scores = [word_score(vocab.decode(s[args.prompt_length :]), known) for s in samples]
        if scores[0] != scores[1]:
            winner, loser = (samples[0], samples[1]) if scores[0] > scores[1] else (samples[1], samples[0])
            pairs.append((winner, loser))
    return pairs


def evaluate(model, reference, vocab, prompts: list[list[int]], known: set[str], args) -> dict:
    """Mean word score of fresh samples, and the policy's log-ratio to the reference on them (a KL estimate)."""

    rng = random.Random(args.seed + 99)
    samples = [generate_tokens(model, p, length=args.continuation, context_size=model.context_size,
                               temperature=1.0, top_k=None, rng=rng) for p in prompts]
    scores = [word_score(vocab.decode(s[args.prompt_length :]), known) for s in samples]
    with no_grad(), evaluation_mode(model), evaluation_mode(reference):
        ratio = (sequence_log_probs(model, samples, args.prompt_length)
                 - sequence_log_probs(reference, samples, args.prompt_length))
    texts = [vocab.decode(s) for s in samples[:3]]
    return {"score": sum(scores) / len(scores), "kl": sum(ratio.data) / len(samples), "samples": texts}


def compare(policy, reference, pairs, args) -> float:
    """One DPO loss and gradient in PyTorch through the twin; the largest gradient difference."""

    import torch

    from torch_twin import TorchTwin

    winners, losers = [w for w, _ in pairs[:4]], [l for _, l in pairs[:4]]
    for parameter in policy.parameters():
        parameter.zero_grad()
    loss, _ = dpo_loss(policy, reference, winners, losers, args.prompt_length, args.beta)
    loss.backward()
    twin, ref_twin = TorchTwin(policy), TorchTwin(reference)
    twin.training = ref_twin.training = False

    def log_probs(network, sequences):
        inputs = torch.tensor([s[:-1] for s in sequences])
        logits = network(inputs)                                   # every position's next-token logits
        picked = torch.log_softmax(logits, -1).gather(2, torch.tensor([s[1:] for s in sequences])[..., None])[..., 0]
        return picked[:, args.prompt_length - 1 :].sum(1)

    with torch.no_grad():
        ref = log_probs(ref_twin, winners + losers)
    pol = log_probs(twin, winners + losers)
    n = len(winners)
    margin = args.beta * ((pol[:n] - pol[n:]) - (ref[:n] - ref[n:]))
    torch_loss = -torch.nn.functional.logsigmoid(margin).mean()
    torch_loss.backward()
    difference = max(abs(a - b) for p, t in zip(policy.parameters(), twin.parameters)
                     for a, b in zip(p.grad, t.grad.reshape(-1).tolist()))
    print(f"DPO loss: NanoGrad {loss.item():.10f}, PyTorch {torch_loss.item():.10f}; "
          f"largest gradient difference {difference:.1e}")
    return difference


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="runs/baby_llama", help="a gpt_train.py --out-dir")
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--continuation", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-prompts", type=int, default=200)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    policy, validation, config = load_run(args.run)
    reference = copy.deepcopy(policy)
    text = Path(config.text_file).read_text(encoding="utf-8")
    vocab = CharVocab.from_text(text)
    train_text, _ = split_text(text, config.validation_fraction)
    known = set(WORD.findall(train_text.lower()))
    training_tokens = vocab.encode(train_text)
    if args.prompt_length + args.continuation > policy.context_size:
        raise ValueError("prompt + continuation must fit in the model's context")

    rng = random.Random(args.seed)
    start = perf_counter()
    pairs = make_pairs(reference, vocab, training_tokens, known, args, rng)
    print(f"{len(pairs)} preference pairs sampled from the model ({perf_counter() - start:.0f} s)", flush=True)
    if args.compare:
        return {"max_gradient_difference": compare(policy, reference, pairs, args)}

    prompts = []
    for _ in range(args.eval_prompts):                    # held-out prompts from the validation text
        at = rng.randrange(len(validation) - args.prompt_length)
        prompts.append(validation[at : at + args.prompt_length])
    before = evaluate(policy, reference, vocab, prompts, known, args)
    print(f"before: word score {before['score']:+.2f} per continuation", flush=True)

    optimizer = TensorAdam(policy.parameters(), lr=args.lr)
    step, start = 0, perf_counter()
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        for first in range(0, len(pairs), args.batch_size):
            batch = pairs[first : first + args.batch_size]
            policy.train()
            optimizer.zero_grad()
            loss, accuracy = dpo_loss(policy, reference, [w for w, _ in batch], [l for _, l in batch],
                                      args.prompt_length, args.beta)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 25 == 0:
                print(f"  step {step:4d}: DPO loss {loss.item():.4f}, pairs ranked right {100 * accuracy:.0f}%, "
                      f"{perf_counter() - start:.0f} s", flush=True)
    after = evaluate(policy, reference, vocab, prompts, known, args)
    print(f"after:  word score {after['score']:+.2f} per continuation (was {before['score']:+.2f}), "
          f"KL to the reference {after['kl']:.2f} nats per continuation", flush=True)
    for text in after["samples"]:
        print("  " + repr(text))
    return {"before": before, "after": after}


if __name__ == "__main__":
    main()
