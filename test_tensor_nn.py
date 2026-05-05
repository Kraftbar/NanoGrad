import unittest

from tensor import Tensor
from tensor_nn import TensorLinear, TensorMLP, binary_mlp, linear


class TensorNNTests(unittest.TestCase):
    def test_tensor_linear_module_uses_parameters(self) -> None:
        layer = TensorLinear(
            inputs=3,
            outputs=2,
            weight=Tensor.from_list([
                [1, 2, 3],
                [4, 5, 6],
            ], requires_grad=True),
            bias=Tensor.from_list([1, -1], requires_grad=True),
        )
        inputs = Tensor.from_list([
            [1, 2, 3],
            [10, 20, 30],
        ])

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(
            outputs.tolist(),
            [
                [15.0, 31.0],
                [141.0, 319.0],
            ],
        )
        self.assertEqual(layer.parameters(), [layer.weight, layer.bias])

    def test_tensor_linear_zero_grad(self) -> None:
        layer = TensorLinear(inputs=2, outputs=1)
        loss = layer(Tensor.from_list([[1, 2]])).sum()
        loss.backward()

        self.assertTrue(any(grad != 0.0 for grad in layer.weight.grad or []))
        layer.zero_grad()

        self.assertEqual(layer.weight.grad, [0.0, 0.0])
        self.assertEqual(layer.bias.grad, [0.0])

    def test_tensor_mlp_forward_and_parameters(self) -> None:
        model = TensorMLP(inputs=2, layers=[3, 1])
        inputs = Tensor.from_list([
            [0, 1],
            [1, 0],
        ])

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (2, 1))
        self.assertEqual(len(model.parameters()), 4)

    def test_tensor_mlp_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            TensorMLP(inputs=2, layers=[])

        with self.assertRaises(ValueError):
            TensorLinear(inputs=0, outputs=1)

        with self.assertRaises(ValueError):
            TensorLinear(
                inputs=2,
                outputs=1,
                weight=Tensor.from_list([[1, 2], [3, 4]]),
            )

    def test_binary_mlp_vector_input(self) -> None:
        hidden_weight = Tensor.from_list([
            [1, -1],
            [-1, 1],
        ])
        hidden_bias = Tensor.from_list([0, 0])
        output_weight = Tensor.from_list([[10, 10]])
        output_bias = Tensor.from_list([-5])

        outputs = binary_mlp(
            Tensor.from_list([1, 0]),
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )

        self.assertEqual(outputs.shape, (1,))
        self.assertGreater(outputs[0], 0.99)

    def test_binary_mlp_batch_input(self) -> None:
        inputs = Tensor.from_list([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        hidden_weight = Tensor.from_list([
            [1, -1],
            [-1, 1],
        ])
        hidden_bias = Tensor.from_list([0, 0])
        output_weight = Tensor.from_list([[10, 10]])
        output_bias = Tensor.from_list([-5])

        outputs = binary_mlp(
            inputs,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )

        self.assertEqual(outputs.shape, (4, 1))
        self.assertLess(outputs[0, 0], 0.01)
        self.assertGreater(outputs[1, 0], 0.99)
        self.assertGreater(outputs[2, 0], 0.99)
        self.assertLess(outputs[3, 0], 0.01)

    def test_linear_vector_input(self) -> None:
        inputs = Tensor.from_list([10, 20, 30])
        weight = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        bias = Tensor.from_list([1, -1])

        outputs = linear(inputs, weight, bias)

        self.assertEqual(outputs.shape, (2,))
        self.assertEqual(outputs.tolist(), [141.0, 319.0])

    def test_linear_batch_input(self) -> None:
        inputs = Tensor.from_list([
            [1, 2, 3],
            [10, 20, 30],
        ])
        weight = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        bias = Tensor.from_list([1, -1])

        outputs = linear(inputs, weight, bias)

        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(
            outputs.tolist(),
            [
                [15.0, 31.0],
                [141.0, 319.0],
            ],
        )

    def test_linear_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            linear(
                Tensor.from_list([1, 2]),
                Tensor.from_list([1, 2]),
                Tensor.from_list([0]),
            )

        with self.assertRaises(ValueError):
            linear(
                Tensor.from_list([1, 2]),
                Tensor.from_list([[1, 2]]),
                Tensor.from_list([[0]]),
            )

        with self.assertRaises(ValueError):
            linear(
                Tensor.from_list([[1, 2]]),
                Tensor.from_list([[1, 2], [3, 4]]),
                Tensor.from_list([0]),
            )


if __name__ == "__main__":
    unittest.main()
