import importlib.util
import random
import unittest

from gradcheck import max_gradient_error
from rnn import LSTM, CharLSTMModel
from tensor import Tensor


def random_tensor(shape, seed, requires_grad=True):
    rng = random.Random(seed)
    count = 1
    for dim in shape:
        count *= dim
    return Tensor([rng.uniform(-1.0, 1.0) for _ in range(count)], shape, requires_grad=requires_grad)


class LSTMTests(unittest.TestCase):
    def test_shapes(self) -> None:
        self.assertEqual(LSTM(4, 6)(random_tensor((3, 5, 4), 1)).shape, (3, 5, 6))
        model = CharLSTMModel(11, embedding_dim=8, hidden=12, num_layers=2)
        self.assertEqual(model.sequence_logits([[1, 2, 3], [4, 5, 6]]).shape, (2, 3, 11))
        self.assertEqual(model([[1, 2, 3], [4, 5, 6]]).shape, (2, 11))

    def test_gradients_match_finite_differences(self) -> None:
        layer = LSTM(3, 4, seed=2)
        inputs = random_tensor((2, 4, 3), 3)
        weights = random_tensor((2, 4, 4), 4, requires_grad=False)
        error = max_gradient_error(lambda: (layer(inputs) * weights).sum(), [inputs, *layer.parameters()])
        self.assertLess(error, 1e-5)

    def test_state_round_trip(self) -> None:
        model, copy = CharLSTMModel(9, embedding_dim=6, hidden=8, seed=1), CharLSTMModel(9, embedding_dim=6, hidden=8, seed=2)
        copy.load_state_dict(model.state_dict())
        self.assertEqual(list(model([[1, 2, 3]]).data), list(copy([[1, 2, 3]]).data))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_matches_pytorch_lstm(self) -> None:
        import torch

        batch, length, inputs_size, hidden = 3, 5, 4, 6
        layer = LSTM(inputs_size, hidden, seed=1)
        inputs = random_tensor((batch, length, inputs_size), 5)
        out = layer(inputs)
        weights = [random.Random(6).uniform(-1.0, 1.0) for _ in range(out.numel)]
        (out * Tensor(weights, out.shape)).sum().backward()

        double = torch.float64
        reference = torch.nn.LSTM(inputs_size, hidden, batch_first=True).double()
        with torch.no_grad():
            reference.weight_ih_l0.copy_(torch.tensor(list(layer.input_weight.data), dtype=double).reshape(4 * hidden, inputs_size))
            reference.weight_hh_l0.copy_(torch.tensor(list(layer.hidden_weight.data), dtype=double).reshape(4 * hidden, hidden))
            reference.bias_ih_l0.copy_(torch.tensor(list(layer.bias.data), dtype=double))
            reference.bias_hh_l0.zero_()
        torch_inputs = torch.tensor(list(inputs.data), dtype=double).reshape(batch, length, inputs_size).requires_grad_()
        torch_out, _ = reference(torch_inputs)
        (torch_out * torch.tensor(weights, dtype=double).reshape(torch_out.shape)).sum().backward()

        for ours, theirs in (
            (out.data, torch_out.detach()),
            (inputs.grad, torch_inputs.grad),
            (layer.hidden_weight.grad, reference.weight_hh_l0.grad),
            (layer.input_weight.grad, reference.weight_ih_l0.grad),
            (layer.bias.grad, reference.bias_ih_l0.grad),
        ):
            for a, b in zip(ours, theirs.reshape(-1).tolist()):
                self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
