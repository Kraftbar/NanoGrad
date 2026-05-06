import tempfile
import unittest
from pathlib import Path

from language import CausalSelfAttention, TransformerBlock, _causal_attention_mask
from tensor import Tensor


class LanguageModuleTests(unittest.TestCase):
    def test_causal_attention_mask_blocks_future_and_cross_batch_tokens(self) -> None:
        mask = _causal_attention_mask(batch_size=2, context_size=3)

        self.assertEqual(mask.shape, (6, 6))
        self.assertEqual(mask[0, 0], 0.0)
        self.assertEqual(mask[0, 1], -1e9)
        self.assertEqual(mask[1, 0], 0.0)
        self.assertEqual(mask[1, 2], -1e9)
        self.assertEqual(mask[3, 3], 0.0)
        self.assertEqual(mask[3, 0], -1e9)

    def test_causal_self_attention_forward_shape(self) -> None:
        attention = CausalSelfAttention(embedding_dim=2, context_size=3, seed=0)
        inputs = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            [
                [0.0, 0.5],
                [1.0, 0.5],
                [0.2, -0.3],
            ],
        ])

        outputs = attention(inputs)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(attention.num_parameters(), 24)

    def test_causal_self_attention_does_not_use_future_tokens(self) -> None:
        attention = CausalSelfAttention(embedding_dim=2, context_size=3, seed=0)
        base = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
        ])
        changed_future = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [10.0, -10.0],
            ],
        ])

        base_outputs = attention(base)
        changed_outputs = attention(changed_future)

        for actual, expected in zip(base_outputs.data[:4], changed_outputs.data[:4]):
            self.assertAlmostEqual(actual, expected)

    def test_causal_self_attention_accumulates_gradients(self) -> None:
        attention = CausalSelfAttention(embedding_dim=2, context_size=3, seed=0)
        inputs = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
        ], requires_grad=True)

        loss = attention(inputs).sum()
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(any(abs(value) > 0.0 for value in inputs.grad or []))
        for parameter in attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(any(abs(value) > 0.0 for value in parameter.grad or []))

    def test_causal_self_attention_state_dict_round_trip(self) -> None:
        attention = CausalSelfAttention(embedding_dim=2, context_size=3, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attention.json"
            attention.save(path)
            loaded = CausalSelfAttention(embedding_dim=2, context_size=3, seed=1)
            loaded.load(path)

        self.assertEqual(loaded.state_dict(), attention.state_dict())

    def test_causal_self_attention_errors(self) -> None:
        with self.assertRaises(ValueError):
            CausalSelfAttention(embedding_dim=0, context_size=1)

        with self.assertRaises(ValueError):
            CausalSelfAttention(embedding_dim=1, context_size=0)

        attention = CausalSelfAttention(embedding_dim=2, context_size=3)
        with self.assertRaises(ValueError):
            attention(Tensor.from_list([[1.0, 2.0]]))

        with self.assertRaises(ValueError):
            attention(Tensor.zeros((1, 2, 2)))

        with self.assertRaises(ValueError):
            attention(Tensor.zeros((1, 3, 1)))

        with self.assertRaises(ValueError):
            attention.load_state_dict({
                "embedding_dim": 3,
                "context_size": 3,
            })

        with self.assertRaises(ValueError):
            attention.load_state_dict({
                "embedding_dim": 2,
                "context_size": 2,
            })

    def test_transformer_block_forward_shape(self) -> None:
        block = TransformerBlock(
            embedding_dim=2,
            context_size=3,
            hidden_dim=4,
            seed=0,
        )
        inputs = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            [
                [0.0, 0.5],
                [1.0, 0.5],
                [0.2, -0.3],
            ],
        ])

        outputs = block(inputs)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(block.num_parameters(), 54)

    def test_transformer_block_does_not_use_future_tokens(self) -> None:
        block = TransformerBlock(
            embedding_dim=2,
            context_size=3,
            hidden_dim=4,
            seed=0,
        )
        base = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
        ])
        changed_future = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [10.0, -10.0],
            ],
        ])

        base_outputs = block(base)
        changed_outputs = block(changed_future)

        for actual, expected in zip(base_outputs.data[:4], changed_outputs.data[:4]):
            self.assertAlmostEqual(actual, expected)

    def test_transformer_block_accumulates_gradients(self) -> None:
        block = TransformerBlock(
            embedding_dim=2,
            context_size=3,
            hidden_dim=4,
            seed=0,
        )
        inputs = Tensor.from_list([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
        ], requires_grad=True)

        loss = block(inputs).sum()
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(any(abs(value) > 0.0 for value in inputs.grad or []))
        self.assertTrue(
            any(parameter.grad is not None for parameter in block.parameters()),
        )

    def test_transformer_block_state_dict_round_trip(self) -> None:
        block = TransformerBlock(
            embedding_dim=2,
            context_size=3,
            hidden_dim=4,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block.json"
            block.save(path)
            loaded = TransformerBlock(
                embedding_dim=2,
                context_size=3,
                hidden_dim=4,
                seed=1,
            )
            loaded.load(path)

        self.assertEqual(loaded.state_dict(), block.state_dict())

    def test_transformer_block_errors(self) -> None:
        with self.assertRaises(ValueError):
            TransformerBlock(embedding_dim=0, context_size=1)

        with self.assertRaises(ValueError):
            TransformerBlock(embedding_dim=1, context_size=0)

        with self.assertRaises(ValueError):
            TransformerBlock(embedding_dim=1, context_size=1, hidden_dim=0)

        block = TransformerBlock(embedding_dim=2, context_size=3, hidden_dim=4)
        with self.assertRaises(ValueError):
            block(Tensor.from_list([[1.0, 2.0]]))

        with self.assertRaises(ValueError):
            block(Tensor.zeros((1, 2, 2)))

        with self.assertRaises(ValueError):
            block(Tensor.zeros((1, 3, 1)))

        with self.assertRaises(ValueError):
            block.load_state_dict({
                "embedding_dim": 3,
                "context_size": 3,
                "hidden_dim": 4,
            })

        with self.assertRaises(ValueError):
            block.load_state_dict({
                "embedding_dim": 2,
                "context_size": 2,
                "hidden_dim": 4,
            })

        with self.assertRaises(ValueError):
            block.load_state_dict({
                "embedding_dim": 2,
                "context_size": 3,
                "hidden_dim": 8,
            })


if __name__ == "__main__":
    unittest.main()
