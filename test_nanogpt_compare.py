import importlib.util
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from nanogpt_compare import (
    CompareConfig,
    NanoGradRunner,
    build_nanograd_model,
    generate,
    matching_prefix_length,
    nanogpt_state_from_nanograd,
    parse_args,
    run,
    split_text,
    training_batches,
    windows,
)
from text import CharVocab


HAS_TORCH = importlib.util.find_spec("torch") is not None


def tiny_config(**overrides) -> CompareConfig:
    values = {
        "context_size": 4,
        "embedding_dim": 8,
        "hidden_dim": 32,
        "heads": 2,
        "layers": 1,
        "lr": 0.01,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 1.0,
        "batch_size": 3,
        "steps": 4,
        "seed": 0,
    }
    values.update(overrides)
    return CompareConfig(**values)


class NanoGPTCompareTests(unittest.TestCase):
    def test_windows_shift_targets_by_one(self) -> None:
        self.assertEqual(
            windows([1, 2, 3, 4, 5], 3),
            [([1, 2, 3], [2, 3, 4]), ([2, 3, 4], [3, 4, 5])],
        )

    def test_split_text_validates_lengths(self) -> None:
        self.assertEqual(split_text("abcdefghij", 4, 2), ("abcdef", "ghij"))
        with self.assertRaises(ValueError):
            split_text("abcdefghij", 2, 2)

    def test_training_batches_are_deterministic(self) -> None:
        pairs = windows(list(range(20)), 3)

        first = training_batches(pairs, batch_size=4, steps=3, seed=7)
        second = training_batches(pairs, batch_size=4, steps=3, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(first[0][0]), 4)

    def test_state_mapping_fuses_attention_heads(self) -> None:
        config = tiny_config()
        model = build_nanograd_model(config, vocab_size=6)

        state = nanogpt_state_from_nanograd(model)

        c_attn = state["transformer.h.0.attn.c_attn.weight"]
        self.assertEqual(len(c_attn), 3 * config.embedding_dim)
        self.assertEqual(len(c_attn[0]), config.embedding_dim)
        attention = model.blocks[0].attention
        self.assertEqual(c_attn[0], attention.queries[0].weight.tolist()[0])
        self.assertEqual(c_attn[4], attention.queries[1].weight.tolist()[0])
        self.assertEqual(c_attn[8], attention.keys[0].weight.tolist()[0])
        self.assertEqual(c_attn[16], attention.values[0].weight.tolist()[0])
        self.assertNotIn("lm_head.weight", state)

    def test_generate_is_reproducible_with_a_seeded_rng(self) -> None:
        vocab = CharVocab.from_text("abc")

        def logits(_context):
            return [0.1, 0.5, 0.2]

        first = generate(
            logits,
            vocab,
            seed_text="ab",
            context_size=2,
            length=10,
            temperature=1.0,
            top_k=2,
            rng=random.Random(3),
        )
        second = generate(
            logits,
            vocab,
            seed_text="ab",
            context_size=2,
            length=10,
            temperature=1.0,
            top_k=2,
            rng=random.Random(3),
        )

        self.assertEqual(first, second)
        self.assertNotIn("a", first[2:])
        self.assertEqual(matching_prefix_length("abcx", "abcy"), 3)

    def test_nanograd_runner_learns(self) -> None:
        config = tiny_config(steps=30)
        text = "abcabcabcabcabcabc"
        vocab = CharVocab.from_text(text)
        pairs = windows(vocab.encode(text), config.context_size)
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        runner = NanoGradRunner(config, len(vocab))

        before = runner.evaluate(xs, ys)
        for _ in range(config.steps):
            runner.train_step(xs, ys)

        self.assertLess(runner.evaluate(xs, ys), before)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_shared_init_matches_nanogpt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args([
                "--preset",
                "tiny-gpt",
                "--steps",
                "6",
                "--eval-every",
                "3",
                "--max-chars",
                "200",
                "--validation-chars",
                "40",
                "--num-samples",
                "2",
                "--generate",
                "20",
                "--output-dir",
                directory,
            ])
            with redirect_stdout(StringIO()):
                result = run(args)

            self.assertTrue((Path(directory) / "samples.csv").exists())
            self.assertTrue((Path(directory) / "manifest.json").exists())

        summary = result["summary"][0]
        self.assertLess(summary["max_validation_loss_gap"], 1e-9)
        self.assertEqual(summary["mean_shared_sample_prefix"], 1.0)


if __name__ == "__main__":
    unittest.main()
