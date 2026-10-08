"""Sequence labeling: a BiLSTM-CRF part-of-speech tagger, NanoGrad vs PyTorch.

Tagging gives every word of a sentence a label: here its universal part of
speech (UPOS: NOUN, VERB, ADP, ... 17 tags), on the English Web Treebank of
Universal Dependencies (EWT: 12.5K training sentences of blogs, reviews,
emails and answers; 2K each to validate and test). The score is the share
of test words tagged right.

The network follows Huang, Xu and Yu (2015) and Lample et al. (2016). Each
word becomes three embeddings joined together: the word itself (lowercased),
its last three letters (suffixes carry much of a part of speech: -ly, -ing,
-ed), and its shape (lowercase, Capitalized, UPPER, or with digits). A
BiLSTM reads the sentence both ways, so every word's vector knows its left
and right context, and a linear layer turns it into one score per tag:
emissions e[t, k].

Picking the best tag for each word on its own ignores that tags depend on
each other (a determiner is rarely followed by a verb). A linear-chain CRF
scores a whole tag sequence y instead:

    score(y) = start[y_1] + sum_t e[t, y_t] + sum_t trans[y_t, y_t+1] + end[y_T]
    p(y | x) = e^score(y) / Z,      Z = sum over every sequence y' of e^score(y')

There are 17^T sequences, but Z comes out of the forward algorithm, one
word at a time: alpha_t[j] is the log of the summed e^score of every
sequence of the first t words that ends in tag j,

    alpha_1[j] = start[j] + e[1, j]
    alpha_t[j] = logsumexp_i(alpha_t-1[i] + trans[i, j]) + e[t, j]
    log Z      = logsumexp_j(alpha_T[j] + end[j])

and the loss is log Z - score(gold tags), whose gradient (through the
logsumexps) is the expected count of each tag and transition minus the
gold count. Decoding (Viterbi) is the same recursion with max instead of
logsumexp, remembering which i won. ``--no-crf`` trains the plain BiLSTM
with a softmax per word, for comparison.

Sentences are batched by length, so a batch has no padding and the math
above holds as written.

    python3 tagger.py --epochs 10
    .venv/bin/python tagger.py --frameworks nanograd torch --epochs 1 --train-limit 400

Data: data/ud_ewt/en_ewt-ud-{train,dev,test}.conllu, from
https://github.com/UniversalDependencies/UD_English-EWT (release r2.13).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

from losses import softmax_cross_entropy
from optim import TensorAdam
from rnn import LSTM
from tensor import Tensor, concat, no_grad
from tensor_nn import TensorEmbedding, TensorLinear, TensorModule

UNKNOWN = 0
SHAPES = ("lower", "Title", "UPPER", "digit")


# -- data ---------------------------------------------------------------------


def read_conllu(path: Path) -> list[tuple[list[str], list[str]]]:
    """(words, UPOS tags) per sentence. Lines of multi-word tokens ("3-4")
    and empty nodes ("8.1") are skipped: the tags belong to the words."""

    sentences, words, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            if words:
                sentences.append((words, tags))
            words, tags = [], []
        elif not line.startswith("#"):
            fields = line.split("\t")
            if fields[0].isdigit():
                words.append(fields[1])
                tags.append(fields[3])
    if words:
        sentences.append((words, tags))
    return sentences


def shape(word: str) -> int:
    if any(ch.isdigit() for ch in word):
        return 3
    if word.isupper() and len(word) > 1:
        return 2
    return 1 if word[:1].isupper() else 0


class Vocabulary:
    """Ids for lowercased words and 3-letter suffixes seen at least twice in
    training (rarer ones share id 0, "unknown"), and for the tags."""

    def __init__(self, sentences: list[tuple[list[str], list[str]]], min_count: int = 2) -> None:
        words = Counter(w.lower() for sentence, _ in sentences for w in sentence)
        suffixes = Counter(w.lower()[-3:] for sentence, _ in sentences for w in sentence)
        self.words = {w: i for i, w in enumerate(sorted(w for w, n in words.items() if n >= min_count), 1)}
        self.suffixes = {s: i for i, s in enumerate(sorted(s for s, n in suffixes.items() if n >= min_count), 1)}
        self.tags = sorted({t for _, tags in sentences for t in tags})
        self.tag_ids = {t: i for i, t in enumerate(self.tags)}

    def encode(self, words: list[str]) -> tuple[list[int], list[int], list[int]]:
        return ([self.words.get(w.lower(), UNKNOWN) for w in words],
                [self.suffixes.get(w.lower()[-3:], UNKNOWN) for w in words],
                [shape(w) for w in words])


def length_batches(examples: list, batch_size: int, rng: random.Random | None) -> list[list]:
    """Batches of sentences of one length (no padding), shuffled if rng is given."""

    by_length: dict[int, list] = {}
    for example in examples:
        by_length.setdefault(len(example[0]), []).append(example)
    batches = []
    for group in by_length.values():
        if rng is not None:
            rng.shuffle(group)
        batches += [group[i : i + batch_size] for i in range(0, len(group), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    return batches


# -- the model ----------------------------------------------------------------


class Tagger(TensorModule):
    """Embeddings -> BiLSTM -> per-tag emissions, and a CRF's transition scores."""

    def __init__(self, vocab: Vocabulary, *, word_dim: int = 100, suffix_dim: int = 30, shape_dim: int = 5,
                 hidden: int = 100, crf: bool = True, seed: int = 0) -> None:
        tags = len(vocab.tags)
        self.word = TensorEmbedding(len(vocab.words) + 1, word_dim, seed=seed)
        self.suffix = TensorEmbedding(len(vocab.suffixes) + 1, suffix_dim, seed=seed + 1)
        self.shape = TensorEmbedding(len(SHAPES), shape_dim, seed=seed + 2)
        inputs = word_dim + suffix_dim + shape_dim
        self.forward_lstm = LSTM(inputs, hidden, seed=seed + 3)
        self.backward_lstm = LSTM(inputs, hidden, seed=seed + 4)
        self.emit = TensorLinear(2 * hidden, tags, seed=seed + 5)
        self.crf = crf
        # trans[i, j]: the score of tag j right after tag i. They start at zero:
        # with no transition scores the CRF begins as the plain tagger.
        self.transitions = Tensor.zeros((tags, tags), requires_grad=True)
        self.start = Tensor.zeros((tags,), requires_grad=True)
        self.end = Tensor.zeros((tags,), requires_grad=True)

    def emissions(self, words: list[list[int]], suffixes: list[list[int]], shapes: list[list[int]]) -> Tensor:
        """e (B, T, tags) for a batch of equal-length sentences."""

        x = concat([self.word(words), self.suffix(suffixes), self.shape(shapes)], axis=2)
        length = x.shape[1]
        forward = self.forward_lstm(x)
        # The backward LSTM reads the sentence right to left; its outputs are
        # flipped back so position t holds what it saw of words t..T.
        backward = reverse_time(self.backward_lstm(reverse_time(x)))
        h = concat([forward, backward], axis=2)                          # (B, T, 2H)
        scores = self.emit(h.reshape((len(words) * length, h.shape[2])))
        return scores.reshape((len(words), length, scores.shape[1]))

    def loss(self, batch) -> Tensor:
        words, suffixes, shapes, tags = (list(column) for column in zip(*batch))
        e = self.emissions(words, suffixes, shapes)
        batch_size, length, count = e.shape
        if not self.crf:
            return softmax_cross_entropy(e.reshape((batch_size * length, count)), [t for row in tags for t in row])
        return (self.log_partition(e) - self.gold_score(e, tags)).sum() * (1.0 / batch_size)

    def log_partition(self, e: Tensor) -> Tensor:
        """log Z per sentence by the forward algorithm (see the module docstring)."""

        batch_size, length, count = e.shape
        steps = [step.reshape((batch_size, count)) for step in e.split([1] * length, axis=1)]
        to_from = self.transitions.T                                       # [j, i] = trans[i, j]
        alpha = steps[0] + self.start
        for step in steps[1:]:
            # scores[b, j, i] = alpha[b, i] + trans[i, j]; sum out the previous tag i
            alpha = (alpha.reshape((batch_size, 1, count)) + to_from).logsumexp(axis=2) + step
        return (alpha + self.end).logsumexp(axis=1)

    def gold_score(self, e: Tensor, tags: list[list[int]]) -> Tensor:
        """score(y) of the given tags, per sentence: emissions picked by a
        one-hot mask, transitions weighted by how often each pair occurs."""

        batch_size, length, count = e.shape
        one_hot = [0.0] * (batch_size * length * count)
        for b, row in enumerate(tags):
            for t, tag in enumerate(row):
                one_hot[(b * length + t) * count + tag] = 1.0
        emitted = (e * Tensor(one_hot, e.shape)).sum(axis=2).sum(axis=1)  # (B,)
        pairs, starts, ends = [0.0] * (count * count), [0.0] * count, [0.0] * count
        for row in tags:
            starts[row[0]] += 1.0
            ends[row[-1]] += 1.0
            for a, b in zip(row, row[1:]):
                pairs[a * count + b] += 1.0
        chain = ((self.transitions * Tensor(pairs, (count, count))).sum() + (self.start * Tensor(starts, (count,))).sum()
                 + (self.end * Tensor(ends, (count,))).sum())
        # The chain part is one number for the whole batch; spread it evenly so
        # the per-sentence scores still sum to the batch total.
        return emitted + chain * (1.0 / batch_size)

    def decode(self, batch) -> list[list[int]]:
        words, suffixes, shapes = (list(column) for column in list(zip(*batch))[:3])
        with no_grad():
            e = self.emissions(words, suffixes, shapes)
        batch_size, length, count = e.shape
        rows = [[list(e.data[(b * length + t) * count : (b * length + t + 1) * count]) for t in range(length)]
                for b in range(batch_size)]
        if not self.crf:
            return [[max(range(count), key=scores.__getitem__) for scores in row] for row in rows]
        trans = [list(self.transitions.data[i * count : (i + 1) * count]) for i in range(count)]
        return [viterbi(row, trans, list(self.start.data), list(self.end.data)) for row in rows]

    def parameters(self) -> list[Tensor]:
        return [*self.word.parameters(), *self.suffix.parameters(), *self.shape.parameters(),
                *self.forward_lstm.parameters(), *self.backward_lstm.parameters(), *self.emit.parameters(),
                self.transitions, self.start, self.end]


def reverse_time(x: Tensor) -> Tensor:
    """(B, T, D) with the time axis reversed."""

    return concat(x.split([1] * x.shape[1], axis=1)[::-1], axis=1)


def viterbi(emissions: list[list[float]], trans: list[list[float]], start: list[float], end: list[float]) -> list[int]:
    """The highest-scoring tag sequence: the forward recursion with max for
    logsumexp, keeping each step's best previous tag to trace back."""

    count = len(start)
    best = [start[j] + emissions[0][j] for j in range(count)]
    back = []
    for scores in emissions[1:]:
        previous = [max(range(count), key=lambda i, j=j: best[i] + trans[i][j]) for j in range(count)]
        best = [best[previous[j]] + trans[previous[j]][j] + scores[j] for j in range(count)]
        back.append(previous)
    tag = max(range(count), key=lambda j: best[j] + end[j])
    path = [tag]
    for previous in reversed(back):
        tag = previous[tag]
        path.append(tag)
    return path[::-1]


# -- one runner per framework ---------------------------------------------------


class NanoGradRunner:
    def __init__(self, model: Tagger, lr: float, clip: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr, max_grad_norm=clip)

    def train_step(self, batch) -> float:
        self.optimizer.zero_grad()
        loss = self.model.loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decode(self, batch) -> list[list[int]]:
        return self.model.decode(batch)

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    """The same tagger in PyTorch, from the NanoGrad model's weights."""

    def __init__(self, model: Tagger, lr: float, clip: float) -> None:
        import torch

        self.torch, self.clip, self.crf = torch, clip, model.crf

        def p(tensor) -> "torch.Tensor":
            return torch.nn.Parameter(torch.tensor(list(tensor.data), dtype=torch.float64).reshape(tensor.shape))

        def lstm(layer: LSTM) -> "torch.nn.LSTM":
            twin = torch.nn.LSTM(layer.inputs, layer.hidden, batch_first=True, dtype=torch.float64)
            with torch.no_grad():
                twin.weight_ih_l0.copy_(p(layer.input_weight))
                twin.weight_hh_l0.copy_(p(layer.hidden_weight))
                twin.bias_ih_l0.copy_(p(layer.bias))
                twin.bias_hh_l0.zero_()                   # NanoGrad's LSTM has the one bias
            twin.bias_hh_l0.requires_grad_(False)
            return twin

        self.word, self.suffix, self.shape = (p(m.weight) for m in (model.word, model.suffix, model.shape))
        self.forward_lstm, self.backward_lstm = lstm(model.forward_lstm), lstm(model.backward_lstm)
        self.emit_weight, self.emit_bias = p(model.emit.weight), p(model.emit.bias)
        self.transitions, self.start, self.end = p(model.transitions), p(model.start), p(model.end)
        self.layout = [self.word, self.suffix, self.shape]
        for twin in (self.forward_lstm, self.backward_lstm):
            self.layout += [twin.weight_ih_l0, twin.weight_hh_l0, twin.bias_ih_l0]
        self.layout += [self.emit_weight, self.emit_bias, self.transitions, self.start, self.end]
        self.optimizer = torch.optim.Adam(self.layout, lr=lr)

    def emissions(self, words, suffixes, shapes):
        torch = self.torch
        x = torch.cat([self.word[torch.tensor(words)], self.suffix[torch.tensor(suffixes)],
                       self.shape[torch.tensor(shapes)]], dim=2)
        forward, _ = self.forward_lstm(x)
        backward, _ = self.backward_lstm(x.flip(1))
        h = torch.cat([forward, backward.flip(1)], dim=2)
        return h @ self.emit_weight.T + self.emit_bias

    def loss(self, batch):
        torch = self.torch
        words, suffixes, shapes, tags = (list(column) for column in zip(*batch))
        e = self.emissions(words, suffixes, shapes)
        tags = torch.tensor(tags)
        if not self.crf:
            return torch.nn.functional.cross_entropy(e.reshape(-1, e.shape[-1]), tags.reshape(-1))
        alpha = e[:, 0] + self.start
        for t in range(1, e.shape[1]):
            alpha = torch.logsumexp(alpha[:, :, None] + self.transitions, dim=1) + e[:, t]
        log_z = torch.logsumexp(alpha + self.end, dim=1)
        gold = (e.gather(2, tags[:, :, None]).squeeze(2).sum(1) + self.start[tags[:, 0]] + self.end[tags[:, -1]]
                + self.transitions[tags[:, :-1], tags[:, 1:]].sum(1))
        return (log_z - gold).mean()

    def train_step(self, batch) -> float:
        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.layout, self.clip)
        self.optimizer.step()
        return loss.item()

    def decode(self, batch) -> list[list[int]]:
        torch = self.torch
        words, suffixes, shapes = (list(column) for column in list(zip(*batch))[:3])
        with torch.no_grad():
            e = self.emissions(words, suffixes, shapes)
        if not self.crf:
            return e.argmax(dim=2).tolist()
        trans, start, end = self.transitions.tolist(), self.start.tolist(), self.end.tolist()
        return [viterbi(row, trans, start, end) for row in e.tolist()]

    def weights(self) -> list[list[float]]:
        return [p.detach().reshape(-1).tolist() for p in self.layout]


def accuracy(runner, examples: list, batch_size: int = 256) -> float:
    right = total = 0
    for batch in length_batches(examples, batch_size, None):
        for predicted, example in zip(runner.decode(batch), batch):
            right += sum(p == t for p, t in zip(predicted, example[3]))
            total += len(predicted)
    return right / total


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/ud_ewt")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--no-crf", dest="crf", action="store_false", help="a softmax per word instead of the CRF")
    parser.add_argument("--train-limit", type=int, help="use only the first N training sentences")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    folder = Path(args.data)
    splits = {name: read_conllu(folder / f"en_ewt-ud-{name}.conllu") for name in ("train", "dev", "test")}
    if args.train_limit:
        splits["train"] = splits["train"][: args.train_limit]
    vocab = Vocabulary(splits["train"])
    examples = {name: [(*vocab.encode(words), [vocab.tag_ids[t] for t in tags]) for words, tags in sentences]
                for name, sentences in splits.items()}
    print(f"{len(examples['train'])} training sentences, {len(vocab.words)} words, {len(vocab.suffixes)} suffixes, "
          f"{len(vocab.tags)} tags; {'CRF' if args.crf else 'softmax'} output", flush=True)

    results, runners = {}, {}
    for framework in args.frameworks:
        model = Tagger(vocab, hidden=args.hidden, crf=args.crf, seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr, args.clip)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr, args.clip)
        print(framework, flush=True)
        rng, start, training, steps = random.Random(args.seed), perf_counter(), 0.0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            total = 0.0
            batches = length_batches(examples["train"], args.batch_size, rng)
            began = perf_counter()
            for batch in batches:
                total += runner.train_step(batch)
            training += perf_counter() - began
            steps += len(batches)
            dev = accuracy(runner, examples["dev"])
            history.append((epoch, total / len(batches), dev))
            print(f"  epoch {epoch:2d}: loss {total / len(batches):.4f}, dev accuracy {100 * dev:.2f}%, "
                  f"{perf_counter() - start:.0f} s", flush=True)
        test = accuracy(runner, examples["test"])
        results[framework] = {"history": history, "dev": history[-1][2], "test": test,
                              "ms_per_step": 1000 * training / max(1, steps)}
        print(f"{framework:8s}: dev {100 * history[-1][2]:.2f}%, test {100 * test:.2f}% "
              f"({results[framework]['ms_per_step']:.1f} ms/step)", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
