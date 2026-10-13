import importlib.util
import random
import unittest

from gradcheck import max_gradient_error
from rnn import GRU, LSTM, CharLSTMModel
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



class GRUTests(unittest.TestCase):
    def test_gradients_match_finite_differences(self) -> None:
        layer = GRU(3, 4, seed=7)
        inputs = random_tensor((2, 4, 3), 8)
        weights = random_tensor((2, 4, 4), 9, requires_grad=False)
        error = max_gradient_error(lambda: (layer(inputs) * weights).sum(), [inputs, *layer.parameters()])
        self.assertLess(error, 1e-5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_matches_pytorch_gru(self) -> None:
        import torch

        batch, length, inputs_size, hidden = 3, 5, 4, 6
        layer = GRU(inputs_size, hidden, seed=10)
        inputs = random_tensor((batch, length, inputs_size), 11)
        out = layer(inputs)
        weights = [random.Random(12).uniform(-1.0, 1.0) for _ in range(out.numel)]
        (out * Tensor(weights, out.shape)).sum().backward()

        double = torch.float64
        reference = torch.nn.GRU(inputs_size, hidden, batch_first=True).double()
        pairs = (
            (reference.weight_ih_l0, layer.input_weight), (reference.weight_hh_l0, layer.hidden_weight),
            (reference.bias_ih_l0, layer.input_bias), (reference.bias_hh_l0, layer.hidden_bias),
        )
        with torch.no_grad():
            for target, source in pairs:
                target.copy_(torch.tensor(list(source.data), dtype=double).reshape(target.shape))
        torch_inputs = torch.tensor(list(inputs.data), dtype=double).reshape(batch, length, inputs_size).requires_grad_()
        torch_out, _ = reference(torch_inputs)
        (torch_out * torch.tensor(weights, dtype=double).reshape(torch_out.shape)).sum().backward()

        for a, b in zip(out.data, torch_out.detach().reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(inputs.grad, torch_inputs.grad.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)
        for target, source in pairs:
            for a, b in zip(source.grad, target.grad.reshape(-1).tolist()):
                self.assertAlmostEqual(a, b, places=12)


def per_step_lstm(layer: LSTM, inputs: Tensor, state):
    """LSTM.run as it was before lstm_sequence: the loop built from per-step ops."""

    from rnn import lstm_memory, lstm_output
    from tensor import concat, matmul

    batch, length, _ = inputs.shape
    input_gates = matmul(inputs, layer.input_weight.T) + layer.bias
    hidden_weight_t = layer.hidden_weight.T
    h, c = state
    outputs = []
    for step in input_gates.split([1] * length, axis=1):
        gates = step.reshape((batch, 4 * layer.hidden)) + matmul(h, hidden_weight_t)
        c = lstm_memory(gates, c)
        h = lstm_output(gates, c)
        outputs.append(h.reshape((batch, 1, layer.hidden)))
    return concat(outputs, axis=1), (h, c)


class SequenceTests(unittest.TestCase):
    def run_both(self, batch, length, inputs_size, hidden, given_state, uses):
        results = []
        for run in (lambda layer, x, state: layer.run(x, state), per_step_lstm):
            layer = LSTM(inputs_size, hidden, seed=7)
            inputs = random_tensor((batch, length, inputs_size), 8)
            state = ((random_tensor((batch, hidden), 9), random_tensor((batch, hidden), 10)) if given_state
                     else (Tensor.zeros((batch, hidden)), Tensor.zeros((batch, hidden))))
            out, (h, c) = run(layer, inputs, state)
            loss = None
            for used, tensor, seed in zip(uses, (out, h, c), (11, 12, 13)):
                if used:
                    term = (tensor * random_tensor(tensor.shape, seed, requires_grad=False)).sum()
                    loss = term if loss is None else loss + term
            loss.backward()
            tensors = [inputs, *layer.parameters(), *(state if given_state else ())]
            results.append(([list(t.data) for t in (out, h, c)], [list(t.grad) for t in tensors]))
        return results

    def test_same_bits_as_the_per_step_ops(self) -> None:
        import kernels

        dtypes = ("float64", "float32") if kernels.BACKEND == "c" else ("float64",)
        before = kernels.default_dtype()
        try:
            for dtype in dtypes:
                kernels.set_default_dtype(dtype)
                for shape in ((3, 5, 4, 6), (2, 1, 3, 9), (5, 7, 6, 20)):
                    for given_state, uses in ((False, (True, False, False)), (True, (True, True, True)),
                                              (False, (False, True, False))):
                        with self.subTest(dtype=dtype, shape=shape, given_state=given_state, uses=uses):
                            fused, steps = self.run_both(*shape, given_state, uses)
                            self.assertEqual(fused, steps)
        finally:
            kernels.set_default_dtype(before)

    def test_reference_kernels_agree(self) -> None:
        import kernels
        import kernels_reference

        if kernels.BACKEND != "c":
            self.skipTest("C kernels are not built here")
        import kernels_c

        rng = random.Random(14)
        batch, steps, hidden = 3, 4, 5
        values = [[rng.uniform(-1, 1) for _ in range(n)]
                  for n in (batch * steps * 4 * hidden, 4 * hidden * hidden, batch * hidden, batch * hidden)]
        grad = [rng.uniform(-1, 1) for _ in range(batch * (steps + 1) * hidden)]
        ours, saved = kernels_c.lstm_sequence(*values, batch, steps, hidden)
        theirs, reference_saved = kernels_reference.lstm_sequence(*values, batch, steps, hidden)
        for a, b in zip(ours, theirs):
            self.assertAlmostEqual(a, b, places=12)
        backward = kernels_c.lstm_sequence_backward(values[1], saved, grad, batch, steps, hidden)
        reference_backward = kernels_reference.lstm_sequence_backward(values[1], reference_saved, grad, batch, steps, hidden)
        for ours_grad, theirs_grad in zip(backward, reference_backward):
            self.assertEqual(len(ours_grad), len(theirs_grad))
            for a, b in zip(ours_grad, theirs_grad):
                self.assertAlmostEqual(a, b, places=12)


class FusedCellTests(unittest.TestCase):
    def test_fused_lstm_ops_match_finite_differences(self) -> None:
        import random

        from rnn import lstm_memory, lstm_output
        from test_gradcheck import max_gradient_error

        rng = random.Random(3)
        gates = Tensor([rng.uniform(-2, 2) for _ in range(2 * 12)], (2, 12), requires_grad=True)
        c = Tensor([rng.uniform(-1, 1) for _ in range(2 * 3)], (2, 3), requires_grad=True)
        weights = Tensor([rng.uniform(-1, 1) for _ in range(6)], (2, 3))

        def loss():
            memory = lstm_memory(gates, c)
            return ((lstm_output(gates, memory) + memory) * weights).sum()

        self.assertLess(max_gradient_error(loss, [gates, c]), 1e-7)


class RowLSTMTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch not installed")
    def test_twin_matches_sequential_mnist_model(self) -> None:
        import random

        import torch

        from rnn import RowLSTM
        from torch_twin import TorchTwin

        model = RowLSTM(rows=5, cols=4, hidden=6, seed=1)
        pixels = [random.Random(2).uniform(0, 1) for _ in range(2 * 20)]
        ours = model(Tensor(pixels, (2, 1, 5, 4)))
        theirs = TorchTwin(model)(torch.tensor(pixels, dtype=torch.float64).reshape(2, 1, 5, 4))
        for a, b in zip(ours.data, theirs.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
