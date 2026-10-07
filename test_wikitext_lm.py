import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from rnn import WordRNNModel, detach_state
from tensor import Tensor
from wikitext_lm import Corpus, NanoGradRunner, batchify, evaluate, parse_args, run, windows


HAS_TORCH = importlib.util.find_spec("torch") is not None


def tiny_corpus(folder: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    sentences = ["the cat sat on a mat", "a dog ran far", "the <unk> sat on the dog"]
    for split, lines in (("train", 200), ("valid", 20), ("test", 20)):
        text = "\n".join(rng.choice(sentences) for _ in range(lines))
        (folder / f"{split}.txt").write_text(text + "\n", encoding="utf-8")


class WikiTextTests(unittest.TestCase):
    def test_corpus_adds_eos_and_shares_one_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for split in ("train", "valid", "test"):
                Path(folder, f"{split}.txt").write_text(" a b \n" if split == "train" else "b c\n", encoding="utf-8")
            corpus = Corpus(folder)
        self.assertEqual(corpus.words, ["a", "b", "<eos>", "c"])
        self.assertEqual(corpus.train, [0, 1, 2])
        self.assertEqual(corpus.test, [1, 3, 2])

    def test_batchify_and_windows_shift_targets_by_one(self) -> None:
        columns = batchify(list(range(23)), 3)                 # 7 per column, 2 dropped
        self.assertEqual(columns, [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20]])
        batches = list(windows(columns, 4))
        self.assertEqual(batches[0], ([[0, 1, 2, 3], [7, 8, 9, 10], [14, 15, 16, 17]],
                                      [[1, 2, 3, 4], [8, 9, 10, 11], [15, 16, 17, 18]]))
        self.assertEqual(batches[1], ([[4, 5], [11, 12], [18, 19]], [[5, 6], [12, 13], [19, 20]]))

    def test_state_carries_across_windows(self) -> None:
        """Reading a text in two windows with the state carried over gives the
        same scores as reading it in one go."""

        model = WordRNNModel(9, embedding_dim=6, hidden=6, num_layers=2, dropout=0.0, seed=2)
        tokens = [[random.Random(3).randrange(9) for _ in range(7)]]
        whole, _ = model(tokens)
        first, state = model([tokens[0][:3]])
        second, _ = model([tokens[0][3:]], detach_state(state))
        joined = list(first.data) + list(second.data)
        for a, b in zip(whole.data, joined):
            self.assertAlmostEqual(a, b, places=12)
        self.assertFalse(detach_state(state)[0][0].requires_grad)

    def test_tied_model_shares_the_embedding(self) -> None:
        model = WordRNNModel(9, embedding_dim=6, hidden=6, tied=True)
        self.assertIs(model.decoder.weight, model.embedding.weight)
        self.assertEqual(len(model.parameters()), 1 + 2 * 3 + 1)
        with self.assertRaises(ValueError):
            WordRNNModel(9, embedding_dim=6, hidden=8, tied=True)

    def test_training_lowers_validation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_corpus(Path(folder))
            args = parse_args(["--data", folder, "--emsize", "8", "--nhid", "8", "--epochs", "3", "--lr", "5",
                               "--batch-size", "4", "--eval-batch-size", "2", "--bptt", "6", "--log-every", "1000"])
            corpus = Corpus(folder)
            from wikitext_lm import build_model

            runner = NanoGradRunner(build_model(len(corpus.words), args), clip=args.clip)
            before = evaluate(runner, batchify(corpus.valid, 2), 6)
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                results = run(args)
        self.assertLess(results["test_ppl"], 2.718281828 ** before)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_pytorch_twin_trains_identically(self) -> None:
        from wikitext_lm import TorchRunner

        for cell, tied in (("lstm", False), ("gru", True)):
            with self.subTest(cell=cell, tied=tied):
                model = WordRNNModel(30, embedding_dim=8, hidden=8, cell=cell, dropout=0.0, tied=tied, seed=1)
                theirs, ours = TorchRunner(model, clip=0.25, dropout=0.0), NanoGradRunner(model, clip=0.25)
                rng = random.Random(0)
                columns = batchify([rng.randrange(30) for _ in range(400)], 4)
                for inputs, targets in list(windows(columns, 5))[:6]:
                    self.assertAlmostEqual(ours.train_step(inputs, targets, 20.0),
                                           theirs.train_step(inputs, targets, 20.0), places=10)
                for a, b in zip(ours.weights(), theirs.weights()):
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x, y, places=10)


if __name__ == "__main__":
    unittest.main()
