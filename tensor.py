"""Small tensor container, shape-checked math helpers, and tensor autograd."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable


Number = int | float


class Tensor:
    """A dense tensor backed by flat row-major data."""

    def __init__(
        self,
        data: Sequence[Number],
        shape: tuple[int, ...],
        *,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        if len(shape) == 0:
            raise ValueError("Tensor shape must have at least one dimension")
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        if len(data) != _numel(shape):
            raise ValueError("data length does not match shape")

        self.data = [float(value) for value in data]
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad: list[float] | None = (
            [0.0] * len(self.data)
            if requires_grad
            else None
        )

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.tolist()})"

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int | tuple[int, ...]) -> float:
        if len(self.shape) == 1:
            if not isinstance(index, int):
                raise IndexError("1D tensors use one integer index")
            return self.data[_normalize_index(index, self.shape[0])]

        if not isinstance(index, tuple) or len(index) != len(self.shape):
            raise IndexError("tensor indices must match tensor dimensions")

        return self.data[_flat_index(index, self.shape)]

    def tolist(self) -> list:
        return _nested_list(self.data, self.shape)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, requires_grad: bool = False) -> "Tensor":
        return cls([0.0] * _numel(shape), shape, requires_grad=requires_grad)

    @classmethod
    def from_list(
        cls,
        values: Sequence,
        *,
        requires_grad: bool = False,
    ) -> "Tensor":
        shape = _infer_shape(values)
        return cls(_flatten_nested(values), shape, requires_grad=requires_grad)

    @classmethod
    def stack(cls, tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        return stack(tensors, axis=axis)

    def __add__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, other)

    def __radd__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, other)

    def __sub__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, -other if isinstance(other, (int, float)) else other * -1)

    def __rsub__(self, other: Number | "Tensor") -> "Tensor":
        return add(-self, other)

    def __neg__(self) -> "Tensor":
        return self * -1

    def __mul__(self, other: Number | "Tensor") -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other: Number | "Tensor") -> "Tensor":
        return multiply(self, other)

    def __truediv__(self, other: Number | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("division by zero")
            return self * (1 / other)
        return self * tensor_reciprocal(other)

    def __rtruediv__(self, other: Number | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            return tensor_reciprocal(self) * other
        return other / self

    def reshape(self, shape: tuple[int, ...]) -> "Tensor":
        return reshape(self, shape)

    def flatten(self, start_axis: int = 0) -> "Tensor":
        return flatten(self, start_axis=start_axis)

    def permute(self, axes: tuple[int, ...]) -> "Tensor":
        return permute(self, axes)

    def conv2d_valid(self, kernel: "Tensor") -> "Tensor":
        return conv2d_valid(self, kernel)

    def avg_pool2d(
        self,
        window: tuple[int, int],
        *,
        stride: tuple[int, int] | None = None,
    ) -> "Tensor":
        return avg_pool2d(self, window, stride=stride)

    def max_pool2d(
        self,
        window: tuple[int, int],
        *,
        stride: tuple[int, int] | None = None,
    ) -> "Tensor":
        return max_pool2d(self, window, stride=stride)

    @property
    def T(self) -> "Tensor":
        return transpose(self)

    def sum(self, axis: int | None = None) -> "Tensor":
        return tensor_sum(self, axis=axis)

    def mean(self, axis: int | None = None) -> "Tensor":
        return tensor_mean(self, axis=axis)

    def softmax(self, axis: int = -1) -> "Tensor":
        return tensor_softmax(self, axis=axis)

    def exp(self) -> "Tensor":
        return tensor_exp(self)

    def log(self) -> "Tensor":
        return tensor_log(self)

    def relu(self) -> "Tensor":
        return tensor_relu(self)

    def sigmoid(self) -> "Tensor":
        return tensor_sigmoid(self)

    def tanh(self) -> "Tensor":
        return tensor_tanh(self)

    def reciprocal(self) -> "Tensor":
        return tensor_reciprocal(self)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = [0.0] * len(self.data)

    def backward(self, grad: Tensor | Sequence[Number] | None = None) -> None:
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)

        if self.grad is None:
            self.grad = [0.0] * len(self.data)
        for node in topo:
            if node.requires_grad:
                node.zero_grad()

        seed = _seed_grad(self, grad)
        for i, value in enumerate(seed):
            self.grad[i] += value

        for node in reversed(topo):
            node._backward()


def add(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise addition with scalar and row-vector broadcasting."""

    if isinstance(right, (int, float)):
        out = Tensor(
            [value + right for value in left.data],
            left.shape,
            requires_grad=left.requires_grad,
            _children=(left,),
            _op="+",
        )

        def _backward() -> None:
            if left.requires_grad:
                _add_grad(left, _grad_data(out))

        out._backward = _backward
        return out

    out_shape = _broadcast_shape(left.shape, right.shape)
    out = Tensor(
        [
            left.data[_broadcast_data_index(index, out_shape, left.shape)]
            + right.data[_broadcast_data_index(index, out_shape, right.shape)]
            for index in range(_numel(out_shape))
        ],
        out_shape,
        requires_grad=left.requires_grad or right.requires_grad,
        _children=(left, right),
        _op="+",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, _sum_broadcast_grad(grad, out_shape, left.shape))
        if right.requires_grad:
            _add_grad(right, _sum_broadcast_grad(grad, out_shape, right.shape))

    out._backward = _backward
    return out


def multiply(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise multiplication with scalar and row-vector broadcasting."""

    if isinstance(right, (int, float)):
        out = Tensor(
            [value * right for value in left.data],
            left.shape,
            requires_grad=left.requires_grad,
            _children=(left,),
            _op="*",
        )

        def _backward() -> None:
            if left.requires_grad:
                _add_grad(left, [right * grad for grad in _grad_data(out)])

        out._backward = _backward
        return out

    out_shape = _broadcast_shape(left.shape, right.shape)
    out = Tensor(
        [
            left.data[_broadcast_data_index(index, out_shape, left.shape)]
            * right.data[_broadcast_data_index(index, out_shape, right.shape)]
            for index in range(_numel(out_shape))
        ],
        out_shape,
        requires_grad=left.requires_grad or right.requires_grad,
        _children=(left, right),
        _op="*",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            left_grad = [0.0] * left.numel
            for index, out_grad in enumerate(grad):
                left_index = _broadcast_data_index(index, out_shape, left.shape)
                right_index = _broadcast_data_index(index, out_shape, right.shape)
                left_grad[left_index] += right.data[right_index] * out_grad
            _add_grad(left, left_grad)
        if right.requires_grad:
            right_grad = [0.0] * right.numel
            for index, out_grad in enumerate(grad):
                left_index = _broadcast_data_index(index, out_shape, left.shape)
                right_index = _broadcast_data_index(index, out_shape, right.shape)
                right_grad[right_index] += left.data[left_index] * out_grad
            _add_grad(right, right_grad)

    out._backward = _backward
    return out


def matmul(left: Tensor, right: Tensor) -> Tensor:
    """Matrix/vector multiplication for 1D and 2D tensors."""

    if len(left.shape) == 1 and len(right.shape) == 1:
        if left.shape[0] != right.shape[0]:
            raise ValueError("vector dot product shapes must match")
        total = sum(
            a * b
            for a, b in zip(left.data, right.data)
        )
        out = Tensor(
            [total],
            (1,),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)[0]
            if left.requires_grad:
                _add_grad(left, [value * grad for value in right.data])
            if right.requires_grad:
                _add_grad(right, [value * grad for value in left.data])

        out._backward = _backward
        return out

    if len(left.shape) == 2 and len(right.shape) == 1:
        rows, shared = left.shape
        if shared != right.shape[0]:
            raise ValueError("matrix-vector inner dimensions must match")

        left_data = left.data
        right_data = right.data
        data = []
        for row in range(rows):
            row_offset = row * shared
            total = 0.0
            for col in range(shared):
                total += left_data[row_offset + col] * right_data[col]
            data.append(total)
        out = Tensor(
            data,
            (rows,),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)
            if left.requires_grad:
                _add_grad(
                    left,
                    [
                        grad[row] * right_data[col]
                        for row in range(rows)
                        for col in range(shared)
                    ],
                )
            if right.requires_grad:
                right_grad = [0.0] * shared
                for row in range(rows):
                    row_offset = row * shared
                    for col in range(shared):
                        right_grad[col] += left_data[row_offset + col] * grad[row]
                _add_grad(right, right_grad)

        out._backward = _backward
        return out

    if len(left.shape) == 1 and len(right.shape) == 2:
        shared = left.shape[0]
        right_rows, cols = right.shape
        if shared != right_rows:
            raise ValueError("vector-matrix inner dimensions must match")

        left_data = left.data
        right_data = right.data
        data = []
        for col in range(cols):
            total = 0.0
            for inner in range(shared):
                total += left_data[inner] * right_data[inner * cols + col]
            data.append(total)
        out = Tensor(
            data,
            (cols,),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)
            if left.requires_grad:
                left_grad = [0.0] * shared
                for inner in range(shared):
                    right_offset = inner * cols
                    for col in range(cols):
                        left_grad[inner] += grad[col] * right_data[right_offset + col]
                _add_grad(left, left_grad)
            if right.requires_grad:
                _add_grad(
                    right,
                    [
                        left_data[inner] * grad[col]
                        for inner in range(shared)
                        for col in range(cols)
                    ],
                )

        out._backward = _backward
        return out

    if len(left.shape) == 2 and len(right.shape) == 2:
        rows, shared = left.shape
        right_rows, cols = right.shape
        if shared != right_rows:
            raise ValueError("matrix-matrix inner dimensions must match")

        left_data = left.data
        right_data = right.data
        data = []
        for row in range(rows):
            left_row_offset = row * shared
            for col in range(cols):
                total = 0.0
                for inner in range(shared):
                    total += (
                        left_data[left_row_offset + inner]
                        * right_data[inner * cols + col]
                    )
                data.append(total)
        out = Tensor(
            data,
            (rows, cols),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)
            if left.requires_grad:
                left_grad = [0.0] * len(left.data)
                for row in range(rows):
                    left_row_offset = row * shared
                    grad_row_offset = row * cols
                    for inner in range(shared):
                        total = 0.0
                        right_row_offset = inner * cols
                        for col in range(cols):
                            total += (
                                grad[grad_row_offset + col]
                                * right_data[right_row_offset + col]
                            )
                        left_grad[left_row_offset + inner] += total
                _add_grad(left, left_grad)
            if right.requires_grad:
                right_grad = [0.0] * len(right.data)
                for inner in range(shared):
                    right_row_offset = inner * cols
                    for col in range(cols):
                        total = 0.0
                        for row in range(rows):
                            total += (
                                left_data[row * shared + inner]
                                * grad[row * cols + col]
                            )
                        right_grad[right_row_offset + col] += total
                _add_grad(right, right_grad)

        out._backward = _backward
        return out

    raise ValueError("matmul expects 1D or 2D tensors")


def reshape(tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Return a tensor view-shaped copy with the same flat data order."""

    if _numel(shape) != len(tensor.data):
        raise ValueError("reshape must preserve the number of elements")

    out = Tensor(
        tensor.data[:],
        shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="reshape",
    )

    def _backward() -> None:
        if tensor.requires_grad:
            _add_grad(tensor, _grad_data(out))

    out._backward = _backward
    return out


def flatten(tensor: Tensor, start_axis: int = 0) -> Tensor:
    """Return tensor values flattened from start_axis onward."""

    start_axis = _normalize_axis(start_axis, tensor.ndim)
    leading_shape = tensor.shape[:start_axis]
    flattened = _numel(tensor.shape[start_axis:])
    return reshape(tensor, (*leading_shape, flattened))


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join same-shaped tensors along a new axis."""

    if not tensors:
        raise ValueError("stack expects at least one tensor")

    base_shape = tensors[0].shape
    for tensor in tensors:
        if tensor.shape != base_shape:
            raise ValueError("stack expects all tensor shapes to match")

    axis = _normalize_insert_axis(axis, len(base_shape))
    out_shape = base_shape[:axis] + (len(tensors),) + base_shape[axis:]
    data = []
    for out_flat_index in range(_numel(out_shape)):
        out_index = _unravel_index(out_flat_index, out_shape)
        tensor_index = out_index[axis]
        source_index = out_index[:axis] + out_index[axis + 1 :]
        data.append(tensors[tensor_index].data[_flat_index(source_index, base_shape)])

    out = Tensor(
        data,
        out_shape,
        requires_grad=any(tensor.requires_grad for tensor in tensors),
        _children=tuple(tensors),
        _op="stack",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        tensor_grads = [
            [0.0] * tensor.numel
            for tensor in tensors
        ]
        for out_flat_index, value in enumerate(grad):
            out_index = _unravel_index(out_flat_index, out_shape)
            tensor_index = out_index[axis]
            source_index = out_index[:axis] + out_index[axis + 1 :]
            tensor_grads[tensor_index][_flat_index(source_index, base_shape)] += value

        for tensor, tensor_grad in zip(tensors, tensor_grads):
            if tensor.requires_grad:
                _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


def permute(tensor: Tensor, axes: tuple[int, ...]) -> Tensor:
    """Return a tensor with axes reordered."""

    axes = _normalize_axes(axes, tensor.ndim)
    out_shape = tuple(tensor.shape[axis] for axis in axes)
    data = []
    for out_flat_index in range(_numel(out_shape)):
        out_index = _unravel_index(out_flat_index, out_shape)
        source_index = [0] * tensor.ndim
        for out_axis, source_axis in enumerate(axes):
            source_index[source_axis] = out_index[out_axis]
        data.append(tensor[tuple(source_index)])

    out = Tensor(
        data,
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="permute",
    )

    def _backward() -> None:
        if not tensor.requires_grad:
            return
        grad = _grad_data(out)
        tensor_grad = [0.0] * tensor.numel
        for out_flat_index, value in enumerate(grad):
            out_index = _unravel_index(out_flat_index, out_shape)
            source_index = [0] * tensor.ndim
            for out_axis, source_axis in enumerate(axes):
                source_index[source_axis] = out_index[out_axis]
            tensor_grad[_flat_index(tuple(source_index), tensor.shape)] += value
        _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


def conv2d_valid(image: Tensor, kernel: Tensor) -> Tensor:
    """Apply a valid convolution-style filter.

    This uses the cross-correlation convention common in neural networks:
    the kernel is slid over the image without flipping it.
    """

    if image.ndim == 2:
        batches = 1
        image_has_batch_axis = False
        input_channels = 1
        image_rows, image_cols = image.shape
        if kernel.ndim == 2:
            output_channels = 1
            kernel_has_output_axis = False
            kernel_channels = 1
            kernel_rows, kernel_cols = kernel.shape
        elif kernel.ndim == 3:
            output_channels, kernel_rows, kernel_cols = kernel.shape
            kernel_has_output_axis = True
            kernel_channels = 1
        else:
            raise ValueError("conv2d_valid kernel shape does not match image shape")
    elif image.ndim == 3:
        batches = 1
        image_has_batch_axis = False
        input_channels, image_rows, image_cols = image.shape
        if kernel.ndim == 3:
            output_channels = 1
            kernel_has_output_axis = False
            kernel_channels, kernel_rows, kernel_cols = kernel.shape
        elif kernel.ndim == 4:
            output_channels, kernel_channels, kernel_rows, kernel_cols = kernel.shape
            kernel_has_output_axis = True
        else:
            raise ValueError("conv2d_valid kernel shape does not match image shape")

        if input_channels != kernel_channels:
            raise ValueError("conv2d_valid image and kernel channels must match")
    elif image.ndim == 4:
        batches, input_channels, image_rows, image_cols = image.shape
        image_has_batch_axis = True
        if kernel.ndim != 4:
            raise ValueError("conv2d_valid kernel shape does not match image shape")
        output_channels, kernel_channels, kernel_rows, kernel_cols = kernel.shape
        kernel_has_output_axis = True

        if input_channels != kernel_channels:
            raise ValueError("conv2d_valid image and kernel channels must match")
    else:
        raise ValueError("conv2d_valid expects a 2D, 3D, or 4D image tensor")

    if kernel_rows > image_rows or kernel_cols > image_cols:
        raise ValueError("conv2d_valid kernel must fit inside the image")

    out_rows = image_rows - kernel_rows + 1
    out_cols = image_cols - kernel_cols + 1
    if image_has_batch_axis:
        out_shape = (batches, output_channels, out_rows, out_cols)
    elif output_channels == 1:
        out_shape = (out_rows, out_cols)
    else:
        out_shape = (output_channels, out_rows, out_cols)

    image_data = image.data
    kernel_data = kernel.data
    image_channel_stride = image_rows * image_cols
    image_batch_stride = input_channels * image_channel_stride
    patch_offsets_by_output = [
        [
            (
                input_channel * image_channel_stride
                + kernel_row * image_cols
                + kernel_col,
                _conv2d_kernel_flat_index(
                    kernel.shape,
                    kernel_has_output_axis,
                    output_channel,
                    input_channel,
                    kernel_row,
                    kernel_col,
                ),
            )
            for input_channel in range(input_channels)
            for kernel_row in range(kernel_rows)
            for kernel_col in range(kernel_cols)
        ]
        for output_channel in range(output_channels)
    ]

    data = []
    for batch in range(batches):
        batch_offset = batch * image_batch_stride
        for output_channel in range(output_channels):
            patch_offsets = patch_offsets_by_output[output_channel]
            for out_row in range(out_rows):
                row_offset = batch_offset + out_row * image_cols
                for out_col in range(out_cols):
                    image_offset = row_offset + out_col
                    total = 0.0
                    for image_step, kernel_index in patch_offsets:
                        total += (
                            image_data[image_offset + image_step]
                            * kernel_data[kernel_index]
                        )
                    data.append(total)

    out = Tensor(
        data,
        out_shape,
        requires_grad=image.requires_grad or kernel.requires_grad,
        _children=(image, kernel),
        _op="conv2d_valid",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        image_grad = [0.0] * len(image.data) if image.requires_grad else None
        kernel_grad = [0.0] * len(kernel.data) if kernel.requires_grad else None
        grad_index = 0

        for batch in range(batches):
            batch_offset = batch * image_batch_stride
            for output_channel in range(output_channels):
                patch_offsets = patch_offsets_by_output[output_channel]
                for out_row in range(out_rows):
                    row_offset = batch_offset + out_row * image_cols
                    for out_col in range(out_cols):
                        image_offset = row_offset + out_col
                        out_grad = grad[grad_index]
                        grad_index += 1

                        for image_step, kernel_index in patch_offsets:
                            image_index = image_offset + image_step
                            if image_grad is not None:
                                image_grad[image_index] += (
                                    out_grad * kernel_data[kernel_index]
                                )
                            if kernel_grad is not None:
                                kernel_grad[kernel_index] += (
                                    out_grad * image_data[image_index]
                                )

        if image.requires_grad:
            assert image_grad is not None
            _add_grad(image, image_grad)
        if kernel.requires_grad:
            assert kernel_grad is not None
            _add_grad(kernel, kernel_grad)

    out._backward = _backward
    return out


def _conv2d_image_value(
    tensor: Tensor,
    has_batch_axis: bool,
    batch: int,
    channel: int,
    row: int,
    col: int,
) -> float:
    if has_batch_axis:
        _, channels, rows, cols = tensor.shape
        return tensor.data[((batch * channels + channel) * rows + row) * cols + col]
    return _conv2d_value(tensor, channel, row, col)


def _conv2d_value(tensor: Tensor, channel: int, row: int, col: int) -> float:
    if tensor.ndim == 2:
        _, cols = tensor.shape
        return tensor.data[row * cols + col]
    _, rows, cols = tensor.shape
    return tensor.data[(channel * rows + row) * cols + col]


def _conv2d_kernel_flat_index(
    shape: tuple[int, ...],
    has_output_axis: bool,
    output_channel: int,
    input_channel: int,
    row: int,
    col: int,
) -> int:
    if len(shape) == 2:
        _, cols = shape
        return row * cols + col
    if len(shape) == 3 and has_output_axis:
        _, rows, cols = shape
        return (output_channel * rows + row) * cols + col
    if len(shape) == 3:
        _, rows, cols = shape
        return (input_channel * rows + row) * cols + col
    _, input_channels, rows, cols = shape
    return ((output_channel * input_channels + input_channel) * rows + row) * cols + col


def _pool2d_flat_index(
    shape: tuple[int, ...],
    batch: int,
    channel: int,
    row: int,
    col: int,
) -> int:
    if len(shape) == 2:
        _, cols = shape
        return row * cols + col
    if len(shape) == 4:
        _, channels, rows, cols = shape
        return ((batch * channels + channel) * rows + row) * cols + col
    _, rows, cols = shape
    return (channel * rows + row) * cols + col


def avg_pool2d(
    image: Tensor,
    window: tuple[int, int],
    *,
    stride: tuple[int, int] | None = None,
) -> Tensor:
    """Average-pool a 2D tensor or each channel in a 3D/4D tensor."""

    if image.ndim == 2:
        batches = 1
        image_has_batch_axis = False
        channels = 1
        image_rows, image_cols = image.shape
        out_shape_prefix = ()
    elif image.ndim == 3:
        batches = 1
        image_has_batch_axis = False
        channels, image_rows, image_cols = image.shape
        out_shape_prefix = (channels,)
    elif image.ndim == 4:
        batches, channels, image_rows, image_cols = image.shape
        image_has_batch_axis = True
        out_shape_prefix = (batches, channels)
    else:
        raise ValueError("avg_pool2d expects a 2D, 3D, or 4D tensor")

    window_rows, window_cols = window
    if window_rows <= 0 or window_cols <= 0:
        raise ValueError("avg_pool2d window dimensions must be positive")

    if stride is None:
        stride_rows, stride_cols = window
    else:
        stride_rows, stride_cols = stride
        if stride_rows <= 0 or stride_cols <= 0:
            raise ValueError("avg_pool2d stride dimensions must be positive")

    if window_rows > image_rows or window_cols > image_cols:
        raise ValueError("avg_pool2d window must fit inside the image")

    out_rows = ((image_rows - window_rows) // stride_rows) + 1
    out_cols = ((image_cols - window_cols) // stride_cols) + 1
    scale = 1 / (window_rows * window_cols)
    data = []
    for batch in range(batches):
        for channel in range(channels):
            for out_row in range(out_rows):
                for out_col in range(out_cols):
                    row_start = out_row * stride_rows
                    col_start = out_col * stride_cols
                    total = 0.0
                    for window_row in range(window_rows):
                        for window_col in range(window_cols):
                            total += _conv2d_image_value(
                                image,
                                image_has_batch_axis,
                                batch,
                                channel,
                                row_start + window_row,
                                col_start + window_col,
                            )
                    data.append(total * scale)

    out = Tensor(
        data,
        (*out_shape_prefix, out_rows, out_cols),
        requires_grad=image.requires_grad,
        _children=(image,),
        _op="avg_pool2d",
    )

    def _backward() -> None:
        if not image.requires_grad:
            return
        grad = _grad_data(out)
        image_grad = [0.0] * len(image.data)
        for batch in range(batches):
            for channel in range(channels):
                for out_row in range(out_rows):
                    for out_col in range(out_cols):
                        row_start = out_row * stride_rows
                        col_start = out_col * stride_cols
                        out_index = _pool2d_flat_index(
                            out.shape,
                            batch,
                            channel,
                            out_row,
                            out_col,
                        )
                        out_grad = grad[out_index] * scale
                        for window_row in range(window_rows):
                            for window_col in range(window_cols):
                                image_index = _pool2d_flat_index(
                                    image.shape,
                                    batch,
                                    channel,
                                    row_start + window_row,
                                    col_start + window_col,
                                )
                                image_grad[image_index] += out_grad
        _add_grad(image, image_grad)

    out._backward = _backward
    return out


def max_pool2d(
    image: Tensor,
    window: tuple[int, int],
    *,
    stride: tuple[int, int] | None = None,
) -> Tensor:
    """Max-pool a 2D tensor or each channel in a 3D/4D tensor.

    If values tie inside a pooling window, the gradient is routed to the first
    maximum in row-major order.
    """

    if image.ndim == 2:
        batches = 1
        channels = 1
        image_rows, image_cols = image.shape
        out_shape_prefix = ()
    elif image.ndim == 3:
        batches = 1
        channels, image_rows, image_cols = image.shape
        out_shape_prefix = (channels,)
    elif image.ndim == 4:
        batches, channels, image_rows, image_cols = image.shape
        out_shape_prefix = (batches, channels)
    else:
        raise ValueError("max_pool2d expects a 2D, 3D, or 4D tensor")

    window_rows, window_cols = window
    if window_rows <= 0 or window_cols <= 0:
        raise ValueError("max_pool2d window dimensions must be positive")

    if stride is None:
        stride_rows, stride_cols = window
    else:
        stride_rows, stride_cols = stride
        if stride_rows <= 0 or stride_cols <= 0:
            raise ValueError("max_pool2d stride dimensions must be positive")

    if window_rows > image_rows or window_cols > image_cols:
        raise ValueError("max_pool2d window must fit inside the image")

    out_rows = ((image_rows - window_rows) // stride_rows) + 1
    out_cols = ((image_cols - window_cols) // stride_cols) + 1
    data = []
    max_indices = []
    for batch in range(batches):
        for channel in range(channels):
            for out_row in range(out_rows):
                for out_col in range(out_cols):
                    row_start = out_row * stride_rows
                    col_start = out_col * stride_cols
                    max_value = None
                    max_index = None
                    for window_row in range(window_rows):
                        for window_col in range(window_cols):
                            image_index = _pool2d_flat_index(
                                image.shape,
                                batch,
                                channel,
                                row_start + window_row,
                                col_start + window_col,
                            )
                            value = image.data[image_index]
                            if max_value is None or value > max_value:
                                max_value = value
                                max_index = image_index
                    assert max_value is not None
                    assert max_index is not None
                    data.append(max_value)
                    max_indices.append(max_index)

    out = Tensor(
        data,
        (*out_shape_prefix, out_rows, out_cols),
        requires_grad=image.requires_grad,
        _children=(image,),
        _op="max_pool2d",
    )

    def _backward() -> None:
        if not image.requires_grad:
            return
        image_grad = [0.0] * len(image.data)
        for out_grad, image_index in zip(_grad_data(out), max_indices):
            image_grad[image_index] += out_grad
        _add_grad(image, image_grad)

    out._backward = _backward
    return out


def transpose(tensor: Tensor) -> Tensor:
    """Transpose a 2D tensor."""

    if len(tensor.shape) != 2:
        raise ValueError("transpose expects a 2D tensor")
    return permute(tensor, (1, 0))


def tensor_sum(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Sum all values or reduce a tensor along one axis."""

    if axis is None:
        out = Tensor(
            [sum(tensor.data)],
            (1,),
            requires_grad=tensor.requires_grad,
            _children=(tensor,),
            _op="sum",
        )

        def _backward() -> None:
            if tensor.requires_grad:
                _add_grad(tensor, [_grad_data(out)[0]] * len(tensor.data))

        out._backward = _backward
        return out

    axis = _normalize_axis(axis, len(tensor.shape))
    out_shape = _reduced_shape(tensor.shape, axis)
    data = [0.0] * _numel(out_shape)

    for flat_index, value in enumerate(tensor.data):
        index = _unravel_index(flat_index, tensor.shape)
        reduced_index = index[:axis] + index[axis + 1 :]
        if not reduced_index:
            reduced_index = (0,)
        data[_flat_index(reduced_index, out_shape)] += value

    out = Tensor(
        data,
        out_shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="sum",
    )

    def _backward() -> None:
        if not tensor.requires_grad:
            return
        grad = _grad_data(out)
        tensor_grad = []
        for flat_index in range(len(tensor.data)):
            index = _unravel_index(flat_index, tensor.shape)
            reduced_index = index[:axis] + index[axis + 1 :]
            if not reduced_index:
                reduced_index = (0,)
            tensor_grad.append(grad[_flat_index(reduced_index, out_shape)])
        _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


def tensor_mean(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Mean of all values or a tensor along one axis."""

    if axis is None:
        return tensor_sum(tensor) * (1 / len(tensor.data))

    axis = _normalize_axis(axis, len(tensor.shape))
    return tensor_sum(tensor, axis=axis) * (1 / tensor.shape[axis])


def tensor_softmax(tensor: Tensor, axis: int = -1) -> Tensor:
    """Softmax over one tensor axis."""

    axis = _normalize_axis(axis, len(tensor.shape))
    groups: dict[tuple[int, ...], list[int]] = {}
    for flat_index in range(tensor.numel):
        index = _unravel_index(flat_index, tensor.shape)
        group_key = index[:axis] + index[axis + 1 :]
        groups.setdefault(group_key, []).append(flat_index)

    data = [0.0] * tensor.numel
    for indices in groups.values():
        values = [tensor.data[index] for index in indices]
        row_max = max(values)
        exp_values = [
            math.exp(value - row_max)
            for value in values
        ]
        exp_sum = sum(exp_values)
        for index, value in zip(indices, exp_values):
            data[index] = value / exp_sum

    out = Tensor(
        data,
        tensor.shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="softmax",
    )

    def _backward() -> None:
        if not tensor.requires_grad:
            return

        grad = _grad_data(out)
        tensor_grad = [0.0] * tensor.numel
        for indices in groups.values():
            weighted_grad = sum(
                grad[index] * out.data[index]
                for index in indices
            )
            for index in indices:
                tensor_grad[index] += out.data[index] * (
                    grad[index] - weighted_grad
                )
        _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


def tensor_exp(tensor: Tensor) -> Tensor:
    """Elementwise exponential."""

    return _map(tensor, math.exp, math.exp, "exp")


def tensor_log(tensor: Tensor) -> Tensor:
    """Elementwise natural logarithm."""

    if any(value <= 0 for value in tensor.data):
        raise ValueError("log is only defined for positive values")
    return _map(tensor, math.log, lambda value: 1 / value, "log")


def tensor_relu(tensor: Tensor) -> Tensor:
    """Elementwise rectified linear unit."""

    return _map(
        tensor,
        lambda value: max(0.0, value),
        lambda value: 1.0 if value > 0 else 0.0,
        "relu",
    )


def tensor_sigmoid(tensor: Tensor) -> Tensor:
    """Elementwise sigmoid."""

    def sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))

    return _map(
        tensor,
        sigmoid,
        lambda value: sigmoid(value) * (1 - sigmoid(value)),
        "sigmoid",
    )


def tensor_tanh(tensor: Tensor) -> Tensor:
    """Elementwise hyperbolic tangent."""

    return _map(
        tensor,
        math.tanh,
        lambda value: 1 - math.tanh(value) ** 2,
        "tanh",
    )


def tensor_reciprocal(tensor: Tensor) -> Tensor:
    """Elementwise reciprocal."""

    if any(value == 0.0 for value in tensor.data):
        raise ValueError("division by zero")
    return _map(
        tensor,
        lambda value: 1 / value,
        lambda value: -1 / (value * value),
        "reciprocal",
    )


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _flat_index(index: tuple[int, ...], shape: tuple[int, ...]) -> int:
    offset = 0
    for value, dim in zip(index, shape):
        offset = offset * dim + _normalize_index(value, dim)
    return offset


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    index = []
    for dim in reversed(shape):
        index.append(flat_index % dim)
        flat_index //= dim
    return tuple(reversed(index))


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("axis out of range")
    return axis


def _normalize_insert_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim + 1
    if axis < 0 or axis > ndim:
        raise ValueError("axis out of range")
    return axis


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if len(axes) != ndim:
        raise ValueError("axes length must match tensor dimensions")

    normalized = tuple(_normalize_axis(axis, ndim) for axis in axes)
    if sorted(normalized) != list(range(ndim)):
        raise ValueError("axes must be a permutation of tensor dimensions")
    return normalized


def _reduced_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    reduced = shape[:axis] + shape[axis + 1 :]
    if not reduced:
        return (1,)
    return reduced


def _infer_shape(values: Sequence) -> tuple[int, ...]:
    if not _is_sequence(values):
        raise ValueError("values must be a non-empty sequence")
    if not values:
        raise ValueError("values must not be empty")

    first = values[0]
    if not _is_sequence(first):
        if any(_is_sequence(value) for value in values):
            raise ValueError("nested input must be rectangular")
        return (len(values),)

    child_shape = _infer_shape(first)
    for value in values:
        if not _is_sequence(value) or _infer_shape(value) != child_shape:
            raise ValueError("nested input must be rectangular")
    return (len(values), *child_shape)


def _flatten_nested(values: Sequence) -> list[float]:
    flat = []
    for value in values:
        if _is_sequence(value):
            flat.extend(_flatten_nested(value))
        else:
            flat.append(float(value))
    return flat


def _nested_list(data: list[float], shape: tuple[int, ...]) -> list:
    if len(shape) == 1:
        return data[: shape[0]]

    step = _numel(shape[1:])
    return [
        _nested_list(data[index * step : (index + 1) * step], shape[1:])
        for index in range(shape[0])
    ]


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _normalize_index(index: int, size: int) -> int:
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError("tensor index out of range")
    return index


def _broadcast_shape(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[int, ...]:
    shape = []
    max_ndim = max(len(left_shape), len(right_shape))
    padded_left = (1,) * (max_ndim - len(left_shape)) + left_shape
    padded_right = (1,) * (max_ndim - len(right_shape)) + right_shape

    for left_dim, right_dim in zip(padded_left, padded_right):
        if left_dim == right_dim:
            shape.append(left_dim)
        elif left_dim == 1:
            shape.append(right_dim)
        elif right_dim == 1:
            shape.append(left_dim)
        else:
            raise ValueError("tensor shapes cannot be broadcast")
    return tuple(shape)


def _broadcast_data_index(
    out_flat_index: int,
    out_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> int:
    out_index = _unravel_index(out_flat_index, out_shape)
    padded_source = (1,) * (len(out_shape) - len(source_shape)) + source_shape
    source_index = [
        0 if source_dim == 1 else out_dim
        for out_dim, source_dim in zip(out_index, padded_source)
    ]
    source_index = source_index[len(source_index) - len(source_shape) :]
    return _flat_index(tuple(source_index), source_shape)


def _sum_broadcast_grad(
    grad: list[float],
    out_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> list[float]:
    source_grad = [0.0] * _numel(source_shape)
    for index, value in enumerate(grad):
        source_grad[_broadcast_data_index(index, out_shape, source_shape)] += value
    return source_grad


def _require_same_shape(left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError("tensor shapes must match")


def _map(tensor: Tensor, fn, grad_fn, op: str) -> Tensor:
    out = Tensor(
        [fn(value) for value in tensor.data],
        tensor.shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op=op,
    )

    def _backward() -> None:
        if tensor.requires_grad:
            _add_grad(
                tensor,
                [
                    grad_fn(value) * grad
                    for value, grad in zip(tensor.data, _grad_data(out))
                ],
            )

    out._backward = _backward
    return out


def _seed_grad(
    tensor: Tensor,
    grad: Tensor | Sequence[Number] | None,
) -> list[float]:
    if grad is None:
        if tensor.shape != (1,):
            raise ValueError("non-scalar tensor backward requires an explicit gradient")
        return [1.0]

    if isinstance(grad, Tensor):
        if grad.shape != tensor.shape:
            raise ValueError("backward gradient shape must match tensor shape")
        return grad.data[:]

    if len(grad) != len(tensor.data):
        raise ValueError("backward gradient length must match tensor data")
    return [float(value) for value in grad]


def _grad_data(tensor: Tensor) -> list[float]:
    if tensor.grad is None:
        return [0.0] * len(tensor.data)
    return tensor.grad


def _add_grad(tensor: Tensor, grad: Sequence[Number]) -> None:
    if tensor.grad is None:
        tensor.grad = [0.0] * len(tensor.data)
    if len(grad) != len(tensor.grad):
        raise ValueError("gradient length does not match tensor data")
    for i, value in enumerate(grad):
        tensor.grad[i] += float(value)
