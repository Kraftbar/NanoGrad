import unittest

from tensor import Tensor
from vision import (
    LeNetishCNN,
    MNISTCNN,
    SimpleCNN,
    ThreeConvCNN,
    TwoConvCNN,
    _activate,
)


class VisionModelTests(unittest.TestCase):
    def test_mnist_cnn_alias_points_to_simple_cnn(self) -> None:
        self.assertIs(MNISTCNN, SimpleCNN)

    def test_lenetish_cnn_alias_points_to_two_conv_cnn(self) -> None:
        self.assertIs(LeNetishCNN, TwoConvCNN)

    def test_simple_cnn_forward_shape(self) -> None:
        model = SimpleCNN(
            image_shape=(1, 2, 2),
            classes=2,
            filters=3,
            kernel_size=2,
            pool_size=1,
            seed=0,
        )

        logits = model(Tensor.from_list([
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                ],
            ],
        ]))

        self.assertEqual(logits.shape, (2, 2))

    def test_simple_cnn_accepts_tanh_activation(self) -> None:
        model = SimpleCNN(
            image_shape=(1, 2, 2),
            classes=2,
            filters=3,
            kernel_size=2,
            pool_size=1,
            activation="tanh",
            seed=0,
        )

        logits = model(Tensor.from_list([
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
        ]))

        self.assertEqual(logits.shape, (1, 2))

    def test_simple_cnn_accepts_max_pooling(self) -> None:
        model = SimpleCNN(
            image_shape=(1, 4, 4),
            classes=2,
            filters=2,
            kernel_size=2,
            pool_size=2,
            pooling="max",
            seed=0,
        )

        logits = model(Tensor.zeros((1, 1, 4, 4)))

        self.assertEqual(model.pooling, "max")
        self.assertEqual(logits.shape, (1, 2))

    def test_two_conv_cnn_forward_shape(self) -> None:
        model = TwoConvCNN(
            image_shape=(1, 28, 28),
            classes=10,
            filters=6,
            second_filters=16,
            kernel_size=5,
            pool_size=2,
            activation="tanh",
            seed=0,
        )

        logits = model(Tensor.zeros((2, 1, 28, 28)))

        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(len(model.parameters()), 6)
        self.assertEqual(model.num_parameters(), 5142)

    def test_two_conv_cnn_accepts_max_pooling(self) -> None:
        model = TwoConvCNN(
            image_shape=(1, 28, 28),
            classes=10,
            filters=6,
            second_filters=16,
            kernel_size=5,
            pool_size=2,
            pooling="max",
            seed=0,
        )

        logits = model(Tensor.zeros((1, 1, 28, 28)))

        self.assertEqual(model.pooling, "max")
        self.assertEqual(logits.shape, (1, 10))

    def test_three_conv_cnn_forward_shape(self) -> None:
        model = ThreeConvCNN(
            image_shape=(3, 32, 32),
            classes=10,
            filters=4,
            second_filters=8,
            third_filters=16,
            kernel_size=3,
            pool_size=2,
            pooling="max",
            seed=0,
        )

        logits = model(Tensor.zeros((2, 3, 32, 32)))

        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(len(model.parameters()), 8)
        self.assertEqual(model.num_parameters(), 2226)

    def test_simple_cnn_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            SimpleCNN(image_shape=(2, 2), classes=2)
        with self.assertRaises(ValueError):
            SimpleCNN(image_shape=(1, 2, 2), classes=0)
        with self.assertRaises(ValueError):
            SimpleCNN(image_shape=(1, 2, 2), classes=2, filters=0)
        with self.assertRaises(ValueError):
            SimpleCNN(image_shape=(1, 2, 2), classes=2, kernel_size=3)
        with self.assertRaises(ValueError):
            SimpleCNN(
                image_shape=(1, 2, 2),
                classes=2,
                kernel_size=2,
                pool_size=2,
            )
        with self.assertRaises(ValueError):
            SimpleCNN(
                image_shape=(1, 2, 2),
                classes=2,
                kernel_size=2,
                activation="sigmoid",
            )
        with self.assertRaises(ValueError):
            SimpleCNN(
                image_shape=(1, 2, 2),
                classes=2,
                kernel_size=2,
                pooling="median",
            )

    def test_two_conv_cnn_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            TwoConvCNN(image_shape=(1, 12, 12), classes=10)

    def test_three_conv_cnn_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            ThreeConvCNN(image_shape=(3, 16, 16), classes=10)

    def test_activation_errors(self) -> None:
        with self.assertRaises(ValueError):
            _activate(Tensor.from_list([1.0]), "sigmoid")


if __name__ == "__main__":
    unittest.main()
