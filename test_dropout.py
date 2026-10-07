import random
import unittest

from gradcheck import max_gradient_error
from language import CharTransformerModel
from tensor import Tensor, dropout, no_grad
from tensor_nn import TensorDropout, TensorMLP
from train import evaluate_tensor_sequence_multiclass_dataset, evaluation_mode, train_tensor_sequence_multiclass_dataset
from datasets import TinySequenceDataset


class DropoutOpTests(unittest.TestCase):
    def test_zeroes_and_rescales(self) -> None:
        values = Tensor([1.0] * 10000, (10000,))

        out = dropout(values, 0.25, random.Random(0))

        kept = [value for value in out.data if value != 0.0]
        self.assertTrue(all(abs(value - 1 / 0.75) < 1e-12 for value in kept))
        self.assertAlmostEqual(len(kept) / 10000, 0.75, delta=0.02)
        # Expected value is preserved.
        self.assertAlmostEqual(sum(out.data) / 10000, 1.0, delta=0.03)

    def test_gradient_flows_only_through_kept_values(self) -> None:
        values = Tensor([0.5, -1.0, 2.0, 3.0, -0.25, 1.5], (2, 3), requires_grad=True)

        # A fresh generator per call keeps the mask fixed, as gradcheck needs.
        error = max_gradient_error(lambda: (dropout(values, 0.5, random.Random(3)) * values).sum(), [values])
        self.assertLess(error, 1e-6)

        out = dropout(values, 0.5, random.Random(3))
        out.sum().backward()
        for value, grad in zip(out.data, values.grad):
            self.assertEqual(grad, 0.0 if value == 0.0 else 2.0)

    def test_zero_probability_is_identity_and_bad_p_raises(self) -> None:
        values = Tensor([1.0, 2.0], (2,))

        self.assertIs(dropout(values, 0.0, random.Random(0)), values)
        for p in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                dropout(values, p, random.Random(0))


class ModeTests(unittest.TestCase):
    def test_module_train_eval_walks_nested_modules(self) -> None:
        model = CharTransformerModel(5, context_size=3, embedding_dim=4, num_layers=2, dropout=0.5)
        dropouts = [module for module in model.modules() if isinstance(module, TensorDropout)]

        self.assertEqual(len(dropouts), 1 + 2 * 3)   # embeddings + 3 per block
        model.eval()
        self.assertTrue(all(not module.training for module in model.modules()))
        model.train()
        self.assertTrue(all(module.training for module in model.modules()))

    def test_eval_mode_is_deterministic_and_matches_no_dropout(self) -> None:
        tokens = Tensor.from_list([[0, 1, 2], [3, 4, 0]])
        with_dropout = CharTransformerModel(5, context_size=3, embedding_dim=4, dropout=0.3, seed=1)
        without = CharTransformerModel(5, context_size=3, embedding_dim=4, dropout=0.0, seed=1)

        with no_grad():
            training_a = with_dropout.sequence_logits(tokens).data
            training_b = with_dropout.sequence_logits(tokens).data
            with evaluation_mode(with_dropout):
                evaluated = with_dropout.sequence_logits(tokens).data
            reference = without.sequence_logits(tokens).data

        self.assertNotEqual(training_a, training_b)   # a new mask every call
        self.assertEqual(evaluated, reference)
        self.assertTrue(with_dropout.training)        # mode restored afterwards

    def test_training_and_evaluation_set_modes(self) -> None:
        xs = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        ys = [[1, 2, 0], [2, 0, 1], [0, 1, 2]]
        dataset = TinySequenceDataset(xs, ys)
        model = CharTransformerModel(3, context_size=3, embedding_dim=4, dropout=0.2, seed=0)
        model.eval()

        train_tensor_sequence_multiclass_dataset(model, dataset, epochs=1, batch_size=3, lr=0.01)
        self.assertTrue(model.training)

        first = evaluate_tensor_sequence_multiclass_dataset(model, dataset).loss
        second = evaluate_tensor_sequence_multiclass_dataset(model, dataset).loss
        self.assertEqual(first, second)
        self.assertTrue(model.training)

    def test_modules_without_dropout_still_have_modes(self) -> None:
        mlp = TensorMLP(2, [3, 1])

        mlp.eval()

        self.assertFalse(mlp.layers[0].training)


if __name__ == "__main__":
    unittest.main()
