"""Run tiny tensor math and tensor-autograd demos."""

from __future__ import annotations

from losses import binary_cross_entropy, mse_loss
from metrics import binary_accuracy, tensor_binary_accuracy
from optim import TensorSGD
from tensor import Tensor, avg_pool2d, matmul
from tensor_nn import TensorConv2D, TensorLinear, TensorMLP, binary_mlp, linear
from train import train_tensor_binary_classifier


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


def convolution_demo() -> None:
    image = Tensor.from_list([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ])
    layer = TensorConv2D(
        (2, 2),
        kernel=Tensor.from_list([
            [1, 0],
            [0, -1],
        ], requires_grad=True),
        bias=Tensor.from_list([0], requires_grad=True),
    )
    features = layer(image)
    pooled = avg_pool2d(features, (2, 2))

    print("\nTensor convolution and pooling")
    print(f"image:    {image.tolist()}")
    print(f"kernel:   {layer.kernel.tolist()}")
    print(f"features: {features.tolist()}")
    print(f"pooled:   {pooled.tolist()}")


def conv_training_demo() -> None:
    layer, history = train_conv_pattern()

    print("\nTensor conv training")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")
    print(f"kernel:       {layer.kernel.tolist()}")
    print(f"bias:         {layer.bias.tolist()}")


def train_conv_pattern(
    *,
    steps: int = 1000,
    lr: float = 0.0005,
) -> tuple[TensorConv2D, list[float]]:
    image = Tensor.from_list([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ])
    target = Tensor.from_list([
        [-5, -5, -5],
        [-5, -5, -5],
        [-5, -5, -5],
    ])
    layer = TensorConv2D(
        (2, 2),
        kernel=Tensor.zeros((2, 2), requires_grad=True),
        bias=Tensor.zeros((1,), requires_grad=True),
    )
    optimizer = TensorSGD(layer.parameters(), lr=lr)
    history = []

    for _ in range(steps):
        prediction = layer(image)
        loss = mse_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss[0])

    return layer, history


def tiny_cnn_classifier_demo() -> None:
    _conv, _classifier, history, accuracy = train_tiny_cnn_classifier()

    print("\nTiny CNN classifier training")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")
    print(f"accuracy:     {accuracy:.3f}")


def train_tiny_cnn_classifier(
    *,
    epochs: int = 400,
    lr: float = 0.1,
) -> tuple[TensorConv2D, TensorLinear, list[float], float]:
    samples = tiny_cnn_samples()
    conv = TensorConv2D((2, 1, 2, 2), seed=0)
    classifier = TensorLinear(inputs=8, outputs=1, seed=1)
    optimizer = TensorSGD([*conv.parameters(), *classifier.parameters()], lr=lr)
    history = []

    for _ in range(epochs):
        total_loss = 0.0
        for image, label in samples:
            probability = tiny_cnn_forward(conv, classifier, image)
            loss = binary_cross_entropy(probability, Tensor.from_list([label]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss[0]
        history.append(total_loss / len(samples))

    probabilities = [
        tiny_cnn_forward(conv, classifier, image)[0]
        for image, _label in samples
    ]
    targets = [
        label
        for _image, label in samples
    ]
    return conv, classifier, history, binary_accuracy(probabilities, targets)


def tiny_cnn_forward(
    conv: TensorConv2D,
    classifier: TensorLinear,
    image: Tensor,
) -> Tensor:
    features = conv(image).relu()
    pooled = avg_pool2d(features, (2, 2), stride=(1, 1))
    return classifier(pooled.flatten()).sigmoid()


def tiny_cnn_samples() -> list[tuple[Tensor, float]]:
    return [
        (
            Tensor.from_list([
                [
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                ],
            ]),
            1.0,
        ),
        (
            Tensor.from_list([
                [
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                ],
            ]),
            1.0,
        ),
        (
            Tensor.from_list([
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                ],
            ]),
            0.0,
        ),
        (
            Tensor.from_list([
                [
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                ],
            ]),
            0.0,
        ),
    ]


def binary_mlp_demo() -> None:
    inputs = Tensor.from_list([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    targets = Tensor.from_list([
        [0],
        [1],
        [1],
        [0],
    ])
    hidden_weight = Tensor.from_list([
        [1, -1],
        [-1, 1],
    ])
    hidden_bias = Tensor.from_list([0, 0])
    output_weight = Tensor.from_list([[10, 10]])
    output_bias = Tensor.from_list([-5])

    probabilities = binary_mlp(
        inputs,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )
    loss = binary_cross_entropy(probabilities, targets)
    accuracy = tensor_binary_accuracy(probabilities, targets)

    print("\nTensor binary MLP")
    print(f"inputs:      {inputs.tolist()}")
    print(f"targets:     {targets.tolist()}")
    print(f"probability: {probabilities.tolist()}")
    print(f"bce:         {loss[0]:.6f}")
    print(f"accuracy:    {accuracy:.3f}")


def tensor_training_demo() -> None:
    inputs = Tensor.from_list([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    targets = Tensor.from_list([
        [0],
        [1],
        [1],
        [0],
    ])
    model = TensorMLP(inputs=2, layers=[2, 1])
    model.layers[0].weight = Tensor.from_list([
        [1.0, -1.0],
        [-1.0, 1.0],
    ], requires_grad=True)
    model.layers[0].bias = Tensor.from_list([0.0, 0.0], requires_grad=True)
    model.layers[1].weight = Tensor.from_list([[0.1, 0.1]], requires_grad=True)
    model.layers[1].bias = Tensor.from_list([0.0], requires_grad=True)

    summary = train_tensor_binary_classifier(
        lambda: model(inputs).sigmoid(),
        targets,
        model.parameters(),
        steps=2000,
        lr=0.5,
    )

    print("\nTensor binary MLP training")
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final loss:   {summary.final_loss:.6f}")
    print(f"accuracy:     {summary.accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")


def main() -> None:
    regression_loss_demo()
    binary_classification_demo()
    matrix_multiply_demo()
    linear_layer_demo()
    convolution_demo()
    conv_training_demo()
    tiny_cnn_classifier_demo()
    binary_mlp_demo()
    tensor_training_demo()


if __name__ == "__main__":
    main()
