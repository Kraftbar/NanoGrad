import tempfile
import unittest
import math
from pathlib import Path

from tensor import Tensor
from tensor_nn import (
    TensorConv2D,
    TensorLinear,
    TensorMLP,
    binary_mlp,
    conv2d_kernel,
    linear,
    xavier_uniform,
)


class TensorNNTests(unittest.TestCase):
    def test_xavier_uniform_initialization(self) -> None:
        weight = xavier_uniform(inputs=3, outputs=2, seed=7)
        repeat = xavier_uniform(inputs=3, outputs=2, seed=7)
        different = xavier_uniform(inputs=3, outputs=2, seed=8)
        limit = math.sqrt(6.0 / (3 + 2))

        self.assertEqual(weight.shape, (2, 3))
        self.assertTrue(weight.requires_grad)
        self.assertTrue(all(-limit <= value <= limit for value in weight.data))
        self.assertEqual(weight.data, repeat.data)
        self.assertNotEqual(weight.data, different.data)

    def test_xavier_uniform_shape_error(self) -> None:
        with self.assertRaises(ValueError):
            xavier_uniform(inputs=0, outputs=1)

    def test_conv2d_kernel_initialization(self) -> None:
        kernel = conv2d_kernel((2, 3), seed=7)
        repeat = conv2d_kernel((2, 3), seed=7)
        different = conv2d_kernel((2, 3), seed=8)
        limit = math.sqrt(6.0 / (6 + 1))

        self.assertEqual(kernel.shape, (2, 3))
        self.assertTrue(kernel.requires_grad)
        self.assertTrue(all(-limit <= value <= limit for value in kernel.data))
        self.assertEqual(kernel.data, repeat.data)
        self.assertNotEqual(kernel.data, different.data)

    def test_conv2d_channel_kernel_initialization(self) -> None:
        kernel = conv2d_kernel((2, 2, 3), seed=7)
        repeat = conv2d_kernel((2, 2, 3), seed=7)
        limit = math.sqrt(6.0 / (12 + 1))

        self.assertEqual(kernel.shape, (2, 2, 3))
        self.assertTrue(kernel.requires_grad)
        self.assertTrue(all(-limit <= value <= limit for value in kernel.data))
        self.assertEqual(kernel.data, repeat.data)

    def test_conv2d_kernel_shape_error(self) -> None:
        with self.assertRaises(ValueError):
            conv2d_kernel((0, 1))
        with self.assertRaises(ValueError):
            conv2d_kernel((1,))

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

    def test_tensor_linear_state_dict_round_trip(self) -> None:
        source = TensorLinear(
            inputs=2,
            outputs=2,
            weight=Tensor.from_list([
                [1, 2],
                [3, 4],
            ], requires_grad=True),
            bias=Tensor.from_list([5, 6], requires_grad=True),
        )
        target = TensorLinear(inputs=2, outputs=2)

        target.load_state_dict(source.state_dict())

        self.assertEqual(target.weight.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(target.bias.tolist(), [5.0, 6.0])
        self.assertTrue(target.weight.requires_grad)
        self.assertTrue(target.bias.requires_grad)

    def test_tensor_conv2d_module_uses_parameters(self) -> None:
        layer = TensorConv2D(
            (2, 2),
            kernel=Tensor.from_list([
                [1, 0],
                [0, -1],
            ], requires_grad=True),
            bias=Tensor.from_list([10], requires_grad=True),
        )
        inputs = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(
            outputs.tolist(),
            [
                [6.0, 6.0],
                [6.0, 6.0],
            ],
        )
        self.assertEqual(layer.parameters(), [layer.kernel, layer.bias])

    def test_tensor_conv2d_module_accepts_channel_stack(self) -> None:
        layer = TensorConv2D(
            (2, 2, 2),
            kernel=Tensor.from_list([
                [
                    [1, 0],
                    [0, -1],
                ],
                [
                    [0.5, 0],
                    [0, -0.5],
                ],
            ], requires_grad=True),
            bias=Tensor.from_list([10], requires_grad=True),
        )
        inputs = Tensor.from_list([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
        ])

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(
            outputs.tolist(),
            [
                [-14.0, -14.0],
                [-14.0, -14.0],
            ],
        )

    def test_tensor_conv2d_zero_grad(self) -> None:
        layer = TensorConv2D((2, 2), seed=0)
        loss = layer(Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])).sum()
        loss.backward()

        self.assertTrue(any(grad != 0.0 for grad in layer.kernel.grad or []))
        self.assertTrue(any(grad != 0.0 for grad in layer.bias.grad or []))
        layer.zero_grad()

        self.assertEqual(layer.kernel.grad, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(layer.bias.grad, [0.0])

    def test_tensor_conv2d_state_dict_round_trip(self) -> None:
        source = TensorConv2D(
            (2, 2),
            kernel=Tensor.from_list([
                [1, 2],
                [3, 4],
            ], requires_grad=True),
            bias=Tensor.from_list([5], requires_grad=True),
        )
        target = TensorConv2D((2, 2))

        target.load_state_dict(source.state_dict())

        self.assertEqual(target.kernel.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(target.bias.tolist(), [5.0])
        self.assertTrue(target.kernel.requires_grad)
        self.assertTrue(target.bias.requires_grad)

    def test_tensor_mlp_forward_and_parameters(self) -> None:
        model = TensorMLP(inputs=2, layers=[3, 1], seed=0)
        repeat = TensorMLP(inputs=2, layers=[3, 1], seed=0)
        inputs = Tensor.from_list([
            [0, 1],
            [1, 0],
        ])

        outputs = model(inputs)

        self.assertEqual(outputs.shape, (2, 1))
        self.assertEqual(len(model.parameters()), 4)
        self.assertEqual(model.state_dict(), repeat.state_dict())

    def test_tensor_mlp_state_dict_round_trip(self) -> None:
        source = TensorMLP(inputs=2, layers=[2, 1])
        source.layers[0].weight.data = [1.0, 2.0, 3.0, 4.0]
        source.layers[0].bias.data = [5.0, 6.0]
        source.layers[1].weight.data = [7.0, 8.0]
        source.layers[1].bias.data = [9.0]
        target = TensorMLP(inputs=2, layers=[2, 1])

        target.load_state_dict(source.state_dict())

        self.assertEqual(target.state_dict(), source.state_dict())

    def test_tensor_mlp_save_and_load(self) -> None:
        source = TensorMLP(inputs=2, layers=[2, 1])
        source.layers[0].weight.data = [1.0, 2.0, 3.0, 4.0]
        source.layers[0].bias.data = [5.0, 6.0]
        source.layers[1].weight.data = [7.0, 8.0]
        source.layers[1].bias.data = [9.0]
        target = TensorMLP(inputs=2, layers=[2, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            source.save(path)
            target.load(path)

        self.assertEqual(target.state_dict(), source.state_dict())

    def test_tensor_state_dict_errors(self) -> None:
        layer = TensorLinear(inputs=2, outputs=1)

        with self.assertRaises(ValueError):
            layer.load_state_dict({})

        with self.assertRaises(ValueError):
            layer.load_state_dict({
                "weight": {
                    "shape": [2, 2],
                    "data": [1, 2, 3, 4],
                },
                "bias": {
                    "shape": [1],
                    "data": [0],
                },
            })

        with self.assertRaises(ValueError):
            TensorMLP(inputs=2, layers=[1]).load_state_dict({"layers": []})

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

        with self.assertRaises(ValueError):
            TensorConv2D((0, 1))

        with self.assertRaises(ValueError):
            TensorConv2D((1,))

        with self.assertRaises(ValueError):
            TensorConv2D(
                (2, 2),
                kernel=Tensor.from_list([[1, 2]]),
            )

        with self.assertRaises(ValueError):
            TensorConv2D(
                (2, 2),
                kernel=Tensor.zeros((1, 2, 2)),
            )

        with self.assertRaises(ValueError):
            TensorConv2D(
                (2, 2),
                bias=Tensor.from_list([1, 2]),
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
