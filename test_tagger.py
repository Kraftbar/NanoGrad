import itertools
import math
import random
import tempfile
import types
import unittest
from pathlib import Path

import tagger
from tensor import Tensor

SAMPLE = """# sent_id = 1
# text = I'll go now.
1-2\tI'll\t_\t_\t_\t_\t_\t_\t_\t_
1\tI\tI\tPRON\tPRP\t_\t3\tnsubj\t_\t_
2\t'll\twill\tAUX\tMD\t_\t3\taux\t_\t_
3\tgo\tgo\tVERB\tVB\t_\t0\troot\t_\t_
4\tnow\tnow\tADV\tRB\t_\t3\tadvmod\t_\t_
4.1\tx\tx\tX\tX\t_\t_\t_\t_\t_
5\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

1\tReally\treally\tADV\tRB\t_\t0\troot\t_\t_
2\t?\t?\tPUNCT\t.\t_\t1\tpunct\t_\t_
"""


def small_model(crf: bool = True, seed: int = 0, chars: bool = False) -> tuple[tagger.Tagger, tagger.Vocabulary]:
    sentences = [(["The", "dog", "runs", "."], ["DET", "NOUN", "VERB", "PUNCT"]),
                 (["A", "cat", "sleeps", "."], ["DET", "NOUN", "VERB", "PUNCT"]),
                 (["Dogs", "run", "."], ["NOUN", "VERB", "PUNCT"])] * 2
    vocab = tagger.Vocabulary(sentences)
    return tagger.Tagger(vocab, word_dim=6, suffix_dim=4, shape_dim=2, hidden=5, crf=crf, chars=chars,
                         char_dim=3, char_hidden=4, seed=seed), vocab


def encoded(vocab: tagger.Vocabulary, words: list[str], tags: list[str]) -> tuple:
    return (*vocab.encode(words), [vocab.tag_ids[t] for t in tags], vocab.spell(words))


def sequence_score(e, trans, start, end, tags) -> float:
    return (start[tags[0]] + sum(e[t][tag] for t, tag in enumerate(tags)) + end[tags[-1]]
            + sum(trans[a][b] for a, b in zip(tags, tags[1:])))


class TaggerTest(unittest.TestCase):
    def test_reads_words_and_tags_skipping_multiword_lines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.conllu"
            path.write_text(SAMPLE, encoding="utf-8")
            sentences = tagger.read_conllu(path)
        self.assertEqual(sentences, [(["I", "'ll", "go", "now", "."], ["PRON", "AUX", "VERB", "ADV", "PUNCT"]),
                                     (["Really", "?"], ["ADV", "PUNCT"])])

    def test_shapes(self) -> None:
        self.assertEqual([tagger.shape(w) for w in ("dog", "Dog", "NASA", "R2D2", "A")], [0, 1, 2, 3, 1])

    def test_spelling_groups(self) -> None:
        groups, rows = tagger.spelling_groups([[(1, 2), (3,), (1, 2)], [(4, 5, 6), (3,), (2, 1)]])
        self.assertEqual(groups, [[(3,)], [(1, 2), (2, 1)], [(4, 5, 6)]])
        table = [w for group in groups for w in group]
        self.assertEqual([table[r] for r in rows], [(1, 2), (3,), (1, 2), (4, 5, 6), (3,), (2, 1)])

    def test_forward_algorithm_and_gold_score_match_brute_force(self) -> None:
        model, _ = small_model()
        rng = random.Random(1)
        count = len(model.start.data)
        for p in (model.transitions, model.start, model.end):
            p.data = [rng.uniform(-1, 1) for _ in p.data]
        batch, length = 2, 3
        values = [rng.uniform(-2, 2) for _ in range(batch * length * count)]
        e = Tensor(values, (batch, length, count))
        log_z = model.log_partition(e).data
        trans = [model.transitions.data[i * count : (i + 1) * count] for i in range(count)]
        tags = [[0, 2, 1], [3, 3, 0]]
        gold = model.gold_score(e, tags).data
        for b in range(batch):
            rows = [values[(b * length + t) * count : (b * length + t + 1) * count] for t in range(length)]
            scores = [sequence_score(rows, trans, model.start.data, model.end.data, seq)
                      for seq in itertools.product(range(count), repeat=length)]
            self.assertAlmostEqual(log_z[b], math.log(sum(math.exp(s) for s in scores)), places=9)
        # gold_score spreads the transition part evenly; the batch total is exact.
        want = sum(sequence_score([values[(b * length + t) * count : (b * length + t + 1) * count]
                                   for t in range(length)], trans, model.start.data, model.end.data, tags[b])
                   for b in range(batch))
        self.assertAlmostEqual(sum(gold), want, places=9)

    def test_viterbi_finds_the_best_sequence(self) -> None:
        rng = random.Random(2)
        count, length = 4, 5
        e = [[rng.uniform(-2, 2) for _ in range(count)] for _ in range(length)]
        trans = [[rng.uniform(-2, 2) for _ in range(count)] for _ in range(count)]
        start, end = [rng.uniform(-1, 1) for _ in range(count)], [rng.uniform(-1, 1) for _ in range(count)]
        best = max(itertools.product(range(count), repeat=length),
                   key=lambda seq: sequence_score(e, trans, start, end, seq))
        self.assertEqual(tagger.viterbi(e, trans, start, end), list(best))

    def test_training_lowers_the_loss(self) -> None:
        model, vocab = small_model()
        examples = [encoded(vocab, ["The", "dog", "runs", "."], ["DET", "NOUN", "VERB", "PUNCT"])]
        runner = tagger.NanoGradRunner(model, 0.05, 5.0)
        first = runner.train_step(examples)
        for _ in range(30):
            last = runner.train_step(examples)
        self.assertLess(last, first / 2)
        self.assertEqual(runner.decode(examples)[0], examples[0][3])

    def test_training_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        for crf, chars in ((True, False), (False, False), (True, True)):
            model, vocab = small_model(crf=crf, chars=chars)
            sentences = [["The", "dog", "runs", "."], ["A", "cat", "sleeps", "."], ["Dogs", "sleep", "now!", "."]]
            tags = [["DET", "NOUN", "VERB", "PUNCT"], ["DET", "NOUN", "VERB", "PUNCT"], ["NOUN", "VERB", "ADV", "PUNCT"]]
            tags = [[t if t in vocab.tag_ids else "NOUN" for t in row] for row in tags]
            batch = [encoded(vocab, w, row) for w, row in zip(sentences, tags)]
            runners = [tagger.NanoGradRunner(model, 1e-2, 5.0), tagger.TorchRunner(model, 1e-2, 5.0)]
            for _ in range(3):
                losses = [runner.train_step(batch) for runner in runners]
                self.assertAlmostEqual(losses[0], losses[1], places=10)
            difference = max(abs(a - b) for x, y in zip(runners[0].weights(), runners[1].weights()) for a, b in zip(x, y))
            self.assertLess(difference, 1e-10)
            self.assertEqual(runners[0].decode(batch), runners[1].decode(batch))



def per_word_log_partition(model, e: Tensor) -> Tensor:
    """Tagger.log_partition as it was before crf_forward: the loop built from per-word ops."""

    batch_size, length, count = e.shape
    steps = [step.reshape((batch_size, count)) for step in e.split([1] * length, axis=1)]
    transition_weights = model.transitions.exp()
    alpha = steps[0] + model.start
    for step in steps[1:]:
        data = alpha.data
        m = Tensor([max(data[b * count : (b + 1) * count]) for b in range(batch_size)], (batch_size, 1))
        alpha = ((alpha - m).exp() @ transition_weights).log() + m + step
    return (alpha + model.end).logsumexp(axis=1)


class ForwardAlgorithmTests(unittest.TestCase):
    def run_both(self, batch, length, count):
        results = []
        for log_partition in (tagger.Tagger.log_partition, per_word_log_partition):
            rng = random.Random(3)
            model = types.SimpleNamespace(
                **{name: Tensor([rng.uniform(-1, 1) for _ in range(size)], shape, requires_grad=True)
                   for name, size, shape in (("transitions", count * count, (count, count)),
                                             ("start", count, (count,)), ("end", count, (count,)))})
            e = Tensor([rng.uniform(-5, 5) for _ in range(batch * length * count)], (batch, length, count),
                       requires_grad=True)
            tags = [[rng.randrange(count) for _ in range(length)] for _ in range(batch)]
            log_z = log_partition(model, e)
            # As in Tagger.loss, so e, start and transitions also get the gold score's gradient.
            loss = (log_z - tagger.Tagger.gold_score(model, e, tags)).sum() * (1.0 / batch)
            loss.backward()
            tensors = (e, model.start, model.end, model.transitions)
            results.append((list(log_z.data), [list(t.grad) for t in tensors]))
        return results

    def test_same_bits_as_the_per_word_ops(self) -> None:
        import kernels

        dtypes = ("float64", "float32") if kernels.BACKEND == "c" else ("float64",)
        before = kernels.default_dtype()
        try:
            for dtype in dtypes:
                kernels.set_default_dtype(dtype)
                for shape in ((3, 1, 5), (2, 2, 17), (4, 6, 17), (32, 9, 17), (1, 4, 1)):
                    with self.subTest(dtype=dtype, shape=shape):
                        fused, per_word = self.run_both(*shape)
                        self.assertEqual(fused, per_word)
        finally:
            kernels.set_default_dtype(before)

    def test_reference_kernels_agree(self) -> None:
        import kernels
        import kernels_reference

        if kernels.BACKEND != "c":
            self.skipTest("C kernels are not built here")
        import kernels_c

        rng = random.Random(4)
        batch, steps, tags = 3, 5, 6
        e = [rng.uniform(-3, 3) for _ in range(batch * steps * tags)]
        start = [rng.uniform(-1, 1) for _ in range(tags)]
        weights = [math.exp(rng.uniform(-1, 1)) for _ in range(tags * tags)]
        grad = [rng.uniform(-1, 1) for _ in range(batch * tags)]
        ours, saved = kernels_c.crf_forward(e, start, weights, batch, steps, tags)
        theirs, reference_saved = kernels_reference.crf_forward(e, start, weights, batch, steps, tags)
        for a, b in zip(ours, theirs, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        backward = kernels_c.crf_forward_backward(weights, saved, grad, batch, steps, tags)
        reference_backward = kernels_reference.crf_forward_backward(weights, reference_saved, grad, batch, steps, tags)
        for ours_grad, theirs_grad in zip(backward, reference_backward):
            self.assertEqual(len(ours_grad), len(theirs_grad))
            for a, b in zip(ours_grad, theirs_grad):
                self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
