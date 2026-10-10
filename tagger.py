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

(each step's logsumexp is a small matrix product of e^alpha and e^trans,
shifted to stay in floating-point range),

and the loss is log Z - score(gold tags), whose gradient (through the
logsumexps) is the expected count of each tag and transition minus the
gold count. Decoding (Viterbi) is the same recursion with max instead of
logsumexp, remembering which i won. ``--no-crf`` trains the plain BiLSTM
with a softmax per word, for comparison.

``--chars`` adds Lample et al.'s character model: a small BiLSTM reads
each word letter by letter (25-dimensional letter embeddings, 25 hidden
units each way), and its two final states join the word's other
embeddings. It sees every prefix and suffix, and spellings of words that
are too rare to have an embedding of their own.

Sentences are batched by length, so a batch has no padding and the math
above holds as written. Dropout (0.5, on the embeddings and on the BiLSTM's
output, as in Lample et al.) keeps the 1.1M weights from memorizing the
training set; the test score reported is the one at the best epoch on dev.

    python3 tagger.py --epochs 10 [--chars]
    .venv/bin/python tagger.py --frameworks nanograd torch --epochs 1 --train-limit 400 --dropout 0

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
from tensor import Tensor, concat, gather_rows, no_grad
from tensor_nn import TensorDropout, TensorEmbedding, TensorLinear, TensorModule
from train import evaluation_mode

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
    training (rarer ones share id 0, "unknown"), for every letter seen in
    training (the rest share 0), and for the tags."""

    def __init__(self, sentences: list[tuple[list[str], list[str]]], min_count: int = 2) -> None:
        words = Counter(w.lower() for sentence, _ in sentences for w in sentence)
        suffixes = Counter(w.lower()[-3:] for sentence, _ in sentences for w in sentence)
        self.words = {w: i for i, w in enumerate(sorted(w for w, n in words.items() if n >= min_count), 1)}
        self.suffixes = {s: i for i, s in enumerate(sorted(s for s, n in suffixes.items() if n >= min_count), 1)}
        self.chars = {c: i for i, c in enumerate(sorted({c for sentence, _ in sentences for w in sentence for c in w}), 1)}
        self.tags = sorted({t for _, tags in sentences for t in tags})
        self.tag_ids = {t: i for i, t in enumerate(self.tags)}

    def encode(self, words: list[str]) -> tuple[list[int], list[int], list[int]]:
        return ([self.words.get(w.lower(), UNKNOWN) for w in words],
                [self.suffixes.get(w.lower()[-3:], UNKNOWN) for w in words],
                [shape(w) for w in words])

    def spell(self, words: list[str]) -> list[tuple[int, ...]]:
        return [tuple(self.chars.get(c, UNKNOWN) for c in w) for w in words]


def spelling_groups(spellings: list[list[tuple[int, ...]]]) -> tuple[list[list[tuple[int, ...]]], list[int]]:
    """The distinct words of a batch, grouped by length (so the character
    LSTM needs no padding), and each word's row in the groups joined end to end."""

    distinct = sorted({w for sentence in spellings for w in sentence}, key=lambda w: (len(w), w))
    groups: dict[int, list] = {}
    for w in distinct:
        groups.setdefault(len(w), []).append(w)
    row = {w: i for i, w in enumerate(distinct)}
    return list(groups.values()), [row[w] for sentence in spellings for w in sentence]


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
                 hidden: int = 100, crf: bool = True, dropout: float = 0.0, chars: bool = False,
                 char_dim: int = 25, char_hidden: int = 25, seed: int = 0) -> None:
        tags = len(vocab.tags)
        self.word = TensorEmbedding(len(vocab.words) + 1, word_dim, seed=seed)
        self.suffix = TensorEmbedding(len(vocab.suffixes) + 1, suffix_dim, seed=seed + 1)
        self.shape = TensorEmbedding(len(SHAPES), shape_dim, seed=seed + 2)
        inputs = word_dim + suffix_dim + shape_dim
        self.chars = chars
        if chars:
            self.char = TensorEmbedding(len(vocab.chars) + 1, char_dim, seed=seed + 7)
            self.char_forward = LSTM(char_dim, char_hidden, seed=seed + 8)
            self.char_backward = LSTM(char_dim, char_hidden, seed=seed + 9)
            inputs += 2 * char_hidden
        self.forward_lstm = LSTM(inputs, hidden, seed=seed + 3)
        self.backward_lstm = LSTM(inputs, hidden, seed=seed + 4)
        self.emit = TensorLinear(2 * hidden, tags, seed=seed + 5)
        self.crf = crf
        self.dropout = dropout
        self.drop = TensorDropout(dropout, seed=seed + 6)         # on the embeddings and the BiLSTM output
        # trans[i, j]: the score of tag j right after tag i. They start at zero:
        # with no transition scores the CRF begins as the plain tagger.
        self.transitions = Tensor.zeros((tags, tags), requires_grad=True)
        self.start = Tensor.zeros((tags,), requires_grad=True)
        self.end = Tensor.zeros((tags,), requires_grad=True)

    def spelled(self, spellings: list[list[tuple[int, ...]]]) -> Tensor:
        """(B, T, 2 * char_hidden): the final states of the character BiLSTM, run once per distinct word."""

        groups, rows = spelling_groups(spellings)
        finals = []
        for group in groups:
            x = self.char([list(w) for w in group])                        # (n, letters, char_dim)
            forward = self.char_forward.run(x)[1][0]
            backward = self.char_backward.run(reverse_time(x))[1][0]
            finals.append(concat([forward, backward], axis=1))
        table = concat(finals, axis=0) if len(finals) > 1 else finals[0]
        return gather_rows(table, rows).reshape((len(spellings), len(spellings[0]), table.shape[1]))

    def emissions(self, words: list[list[int]], suffixes: list[list[int]], shapes: list[list[int]],
                  spellings: list[list[tuple[int, ...]]] | None = None) -> Tensor:
        """e (B, T, tags) for a batch of equal-length sentences."""

        parts = [self.word(words), self.suffix(suffixes), self.shape(shapes)]
        if self.chars:
            parts.append(self.spelled(spellings))
        x = self.drop(concat(parts, axis=2))
        length = x.shape[1]
        forward = self.forward_lstm(x)
        # The backward LSTM reads the sentence right to left; its outputs are
        # flipped back so position t holds what it saw of words t..T.
        backward = reverse_time(self.backward_lstm(reverse_time(x)))
        h = self.drop(concat([forward, backward], axis=2))              # (B, T, 2H)
        scores = self.emit(h.reshape((len(words) * length, h.shape[2])))
        return scores.reshape((len(words), length, scores.shape[1]))

    def loss(self, batch) -> Tensor:
        words, suffixes, shapes, tags, spellings = (list(column) for column in zip(*batch))
        e = self.emissions(words, suffixes, shapes, spellings)
        batch_size, length, count = e.shape
        if not self.crf:
            return softmax_cross_entropy(e.reshape((batch_size * length, count)), [t for row in tags for t in row])
        return (self.log_partition(e) - self.gold_score(e, tags)).sum() * (1.0 / batch_size)

    def log_partition(self, e: Tensor) -> Tensor:
        """log Z per sentence by the forward algorithm (see the module docstring).

        Each step's logsumexp over the previous tag is a matrix product in
        probability space, shifted so the numbers stay in range:

            logsumexp_i(alpha[i] + trans[i, j]) = m + log(sum_i e^(alpha[i] - m) * e^trans[i, j])

        with m the row's largest alpha (any constant works; the largest makes
        every e^(alpha - m) <= 1, with one of them exactly 1).
        """

        batch_size, length, count = e.shape
        steps = [step.reshape((batch_size, count)) for step in e.split([1] * length, axis=1)]
        transition_weights = self.transitions.exp()                        # e^trans[i, j]
        alpha = steps[0] + self.start
        for step in steps[1:]:
            data = alpha.data
            m = Tensor([max(data[b * count : (b + 1) * count]) for b in range(batch_size)], (batch_size, 1))
            alpha = ((alpha - m).exp() @ transition_weights).log() + m + step
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
        words, suffixes, shapes, _, spellings = (list(column) for column in zip(*batch))
        with no_grad(), evaluation_mode(self):
            e = self.emissions(words, suffixes, shapes, spellings)
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
                self.transitions, self.start, self.end, *self.char_parameters()]

    def char_parameters(self) -> list[Tensor]:
        if not self.chars:
            return []
        return [*self.char.parameters(), *self.char_forward.parameters(), *self.char_backward.parameters()]


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
        self.model.train()
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
        self.drop = torch.nn.Dropout(model.dropout)       # its masks differ from NanoGrad's: compare at 0

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
        self.chars = model.chars
        if model.chars:
            self.char = p(model.char.weight)
            self.char_forward, self.char_backward = lstm(model.char_forward), lstm(model.char_backward)
            self.layout.append(self.char)
            for twin in (self.char_forward, self.char_backward):
                self.layout += [twin.weight_ih_l0, twin.weight_hh_l0, twin.bias_ih_l0]
        self.optimizer = torch.optim.Adam(self.layout, lr=lr)

    def spelled(self, spellings):
        torch = self.torch
        groups, rows = spelling_groups(spellings)
        finals = []
        for group in groups:
            x = self.char[torch.tensor([list(w) for w in group])]
            _, (forward, _) = self.char_forward(x)
            _, (backward, _) = self.char_backward(x.flip(1))
            finals.append(torch.cat([forward[0], backward[0]], dim=1))
        table = torch.cat(finals, dim=0)
        return table[torch.tensor(rows)].reshape(len(spellings), len(spellings[0]), table.shape[1])

    def emissions(self, words, suffixes, shapes, spellings):
        torch = self.torch
        parts = [self.word[torch.tensor(words)], self.suffix[torch.tensor(suffixes)], self.shape[torch.tensor(shapes)]]
        if self.chars:
            parts.append(self.spelled(spellings))
        x = self.drop(torch.cat(parts, dim=2))
        forward, _ = self.forward_lstm(x)
        backward, _ = self.backward_lstm(x.flip(1))
        h = self.drop(torch.cat([forward, backward.flip(1)], dim=2))
        return h @ self.emit_weight.T + self.emit_bias

    def loss(self, batch):
        torch = self.torch
        words, suffixes, shapes, tags, spellings = (list(column) for column in zip(*batch))
        e = self.emissions(words, suffixes, shapes, spellings)
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
        self.drop.train()
        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.layout, self.clip)
        self.optimizer.step()
        return loss.item()

    def decode(self, batch) -> list[list[int]]:
        torch = self.torch
        words, suffixes, shapes, _, spellings = (list(column) for column in zip(*batch))
        self.drop.eval()
        with torch.no_grad():
            e = self.emissions(words, suffixes, shapes, spellings)
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
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-crf", dest="crf", action="store_false", help="a softmax per word instead of the CRF")
    parser.add_argument("--chars", action="store_true", help="add a character BiLSTM per word (Lample et al.)")
    parser.add_argument("--train-limit", type=int, help="use only the first N training sentences")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    folder = Path(args.data)
    splits = {name: read_conllu(folder / f"en_ewt-ud-{name}.conllu") for name in ("train", "dev", "test")}
    if args.train_limit:
        splits["train"] = splits["train"][: args.train_limit]
    vocab = Vocabulary(splits["train"])
    examples = {name: [(*vocab.encode(words), [vocab.tag_ids[t] for t in tags], vocab.spell(words))
                       for words, tags in sentences]
                for name, sentences in splits.items()}
    print(f"{len(examples['train'])} training sentences, {len(vocab.words)} words, {len(vocab.suffixes)} suffixes, "
          f"{len(vocab.tags)} tags; {'CRF' if args.crf else 'softmax'} output{', character BiLSTM' if args.chars else ''}", flush=True)

    results, runners = {}, {}
    for framework in args.frameworks:
        model = Tagger(vocab, hidden=args.hidden, crf=args.crf, dropout=args.dropout, chars=args.chars, seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr, args.clip)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr, args.clip)
        print(framework, flush=True)
        rng, start, training, steps = random.Random(args.seed), perf_counter(), 0.0, 0
        history, best = [], (-1.0, 0, 0.0)          # (dev accuracy, epoch, test accuracy then)
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
            if dev > best[0]:                             # the test score of the best epoch on dev
                best = (dev, epoch, accuracy(runner, examples["test"]))
            print(f"  epoch {epoch:2d}: loss {total / len(batches):.4f}, dev accuracy {100 * dev:.2f}%, "
                  f"{perf_counter() - start:.0f} s", flush=True)
        results[framework] = {"history": history, "dev": best[0], "best_epoch": best[1], "test": best[2],
                              "ms_per_step": 1000 * training / max(1, steps)}
        print(f"{framework:8s}: best dev {100 * best[0]:.2f}% (epoch {best[1]}), test then {100 * best[2]:.2f}% "
              f"({results[framework]['ms_per_step']:.1f} ms/step)", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
