import importlib.util
import random
import unittest

from optim import TensorAdam
from seq2seq import EOS, Seq2SeqTransformer, beam_translate, greedy_translate, pad
from translate import batch_loss, bleu, tokenize


HAS_TORCH = importlib.util.find_spec("torch") is not None


class BleuTests(unittest.TestCase):
    def test_perfect_and_empty_overlap(self) -> None:
        sentence = "a man rides a red bike down the hill .".split()
        self.assertAlmostEqual(bleu([sentence], [sentence]), 100.0)
        self.assertEqual(bleu([["x", "y", "z", "w"]], [sentence]), 0.0)

    def test_brevity_penalty(self) -> None:
        reference = "a b c d e f g h".split()
        self.assertAlmostEqual(bleu([reference[:4]], [reference]), 100.0 * 2.718281828459045 ** (1 - 8 / 4), places=6)

    def test_tokenize_keeps_umlauts_and_splits_punctuation(self) -> None:
        self.assertEqual(tokenize("Ein Mädchen läuft."), ["ein", "mädchen", "läuft", "."])


class Seq2SeqTests(unittest.TestCase):
    def test_padding_does_not_change_the_outputs(self) -> None:
        model = Seq2SeqTransformer(12, 10, width=8, heads=2, hidden=16, seed=1)
        source, target = pad([[4, 5, 6, EOS], [7, EOS]]), pad([[1, 3, 4], [1, 5]])
        plain = list(model(source, target).data)
        padded = list(model([row + [0, 0, 0] for row in source], target).data)
        self.assertEqual(plain, padded)

    def test_learns_to_reverse_sequences(self) -> None:
        rng = random.Random(0)
        model = Seq2SeqTransformer(10, 10, context_size=8, width=32, heads=2, hidden=64, seed=2)
        optimizer = TensorAdam(model.parameters(), lr=3e-3)
        for _ in range(250):
            batch = []
            for _ in range(16):
                words = [rng.randrange(3, 10) for _ in range(rng.randrange(2, 5))]
                batch.append((words, words[::-1]))
            optimizer.zero_grad()
            batch_loss(model, batch).backward()
            optimizer.step()
        tests = [[3, 4, 5], [9, 8, 7, 6], [5, 9]]
        outputs = greedy_translate(model, [words + [EOS] for words in tests], max_length=6)
        self.assertEqual([out[:-1] for out in outputs], [words[::-1] for words in tests])
        for words in tests:
            self.assertEqual(beam_translate(model, words + [EOS], beam=3, max_length=6)[:-1], words[::-1])

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_pytorch_twin_computes_the_same_scores_and_gradients(self) -> None:
        import torch

        from tensor import Tensor
        from torch_twin import TorchTwin

        model = Seq2SeqTransformer(20, 15, width=16, heads=2, hidden=32, seed=1)
        source, target = pad([[5, 6, 7, EOS], [8, 9, EOS]]), pad([[1, 4, 5], [1, 6]])
        ours = model(source, target)
        twin = TorchTwin(model)
        theirs = twin.seq2seq(model, source, target)
        for a, b in zip(ours.data, theirs.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)
        weights = torch.randn(theirs.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        (ours * Tensor(weights.reshape(-1).tolist(), ours.shape)).sum().backward()
        (theirs * weights).sum().backward()
        for parameter, tensor in zip(model.parameters(), twin.parameters):
            for a, b in zip(parameter.grad, tensor.grad.reshape(-1).tolist()):
                self.assertAlmostEqual(a, b, places=11)


if __name__ == "__main__":
    unittest.main()
