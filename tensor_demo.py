"""Run a tiny non-autograd tensor math demo."""

from __future__ import annotations

from losses import binary_cross_entropy, mse_loss
from metrics import tensor_binary_accuracy
from tensor import Tensor, matmul
from tensor_nn import linear


def regression_loss_demo() -> None:
    predictions = Tensor.from_list([1, 2, 3])
    targets = Tensor.from_list([1, 4, 5])
    loss = mse_loss(predictions, targets)

    print("Tensor regression loss")
    print(f"predictions: {predictions.tolist()}")
    print(f"targets:     {targets.tolist()}")
    print(f"mse:         {loss[0]:.6f}")


def binary_classification_demo() -> None:
    probabilities = Tensor.from_list([0.1, 0.8, 0.4, 0.7])
    targets = Tensor.from_list([0, 1, 1, 1])
    loss = binary_cross_entropy(probabilities, targets)
    accuracy = tensor_binary_accuracy(probabilities, targets)

    print("\nTensor binary classification")
    print(f"probability: {probabilities.tolist()}")
    print(f"targets:     {targets.tolist()}")
    print(f"bce:         {loss[0]:.6f}")
    print(f"accuracy:    {accuracy:.3f}")


def matrix_multiply_demo() -> None:
    weights = Tensor.from_list([
        [1, 2, 3],
        [4, 5, 6],
    ])
    inputs = Tensor.from_list([10, 20, 30])
    outputs = matmul(weights, inputs)

    print("\nTensor matrix-vector multiply")
    print(f"weights: {weights.tolist()}")
    print(f"inputs:  {inputs.tolist()}")
    print(f"output:  {outputs.tolist()}")


def linear_layer_demo() -> None:
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

    print("\nTensor linear layer")
    print(f"inputs: {inputs.tolist()}")
    print(f"weight: {weight.tolist()}")
    print(f"bias:   {bias.tolist()}")
    print(f"output: {outputs.tolist()}")


def main() -> None:
    regression_loss_demo()
    binary_classification_demo()
    matrix_multiply_demo()
    linear_layer_demo()


if __name__ == "__main__":
    main()
