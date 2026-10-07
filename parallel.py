"""Data-parallel gradients: split each batch across worker processes.

Python runs one thread of bytecode at a time, so to use several CPU cores
NanoGrad uses several *processes*. Each worker keeps its own copy of the
model. For every training step:

1. the main process sends the current weights and one slice of the batch
   to each worker
2. each worker runs forward + backward on its slice
3. the main process averages the workers' gradients, weighted by how many
   targets each slice had, and writes the result into ``parameter.grad``

The loss is a mean over all targets, so this weighted average is *exactly*
the gradient of the whole batch (up to floating-point summation order).
The optimizer step then runs once, in the main process, as usual.

The cost is sending the weights out and the gradients back every step, so
it pays off when a step is much more expensive than copying the model.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sequence

from losses import softmax_cross_entropy
from tensor import Tensor
from tensor_nn import TensorModule


LossFunction = Callable[[TensorModule, list, list], Tensor]


def sequence_loss(model: TensorModule, xs: list, ys: list) -> Tensor:
    """Mean next-token cross entropy over every position of every sequence."""

    logits = model.sequence_logits(Tensor.from_list(xs))
    batch, context, vocab = logits.shape
    targets = [int(target) for row in ys for target in row]
    return softmax_cross_entropy(logits.reshape((batch * context, vocab)), targets)


def target_count(ys: list) -> int:
    """How many targets a slice contributes to the mean loss."""

    return sum(len(row) if isinstance(row, (list, tuple)) else 1 for row in ys)


class DataParallel:
    """Compute a batch's loss and gradients on ``workers`` processes.

    Use as a context manager so the worker processes are always stopped::

        with DataParallel(model, sequence_loss, workers=4) as parallel:
            loss = parallel.loss_and_gradients(xs, ys)   # fills .grad
            optimizer.step()
    """

    def __init__(self, model: TensorModule, loss_fn: LossFunction, *, workers: int) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.model = model
        self.loss_fn = loss_fn
        self.parameters = model.parameters()
        context = multiprocessing.get_context()
        self._connections = []
        self._processes = []
        for _ in range(workers):
            parent_end, child_end = context.Pipe()
            process = context.Process(target=_worker, args=(child_end, model, loss_fn), daemon=True)
            process.start()
            child_end.close()
            self._connections.append(parent_end)
            self._processes.append(process)

    def loss_and_gradients(self, xs: Sequence, ys: Sequence) -> float:
        """Mean loss over the batch; also sets every parameter's ``.grad``."""

        weights = [parameter.data for parameter in self.parameters]
        training = self.model.training
        shards = _split(list(xs), list(ys), len(self._connections))
        busy = []
        for connection, (shard_xs, shard_ys) in zip(self._connections, shards):
            if shard_xs:
                connection.send((weights, training, shard_xs, shard_ys))
                busy.append(connection)

        total_targets = 0
        total_loss = 0.0
        summed = [[0.0] * parameter.numel for parameter in self.parameters]
        for connection in busy:
            loss, gradients, count = connection.recv()
            total_targets += count
            total_loss += loss * count
            for target, gradient in zip(summed, gradients):
                for index, value in enumerate(gradient):
                    target[index] += value * count

        for parameter, gradient in zip(self.parameters, summed):
            parameter.grad = [value / total_targets for value in gradient]
        return total_loss / total_targets

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self._connections = []
        self._processes = []

    def __enter__(self) -> "DataParallel":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _split(xs: list, ys: list, parts: int) -> list[tuple[list, list]]:
    """Contiguous, nearly equal slices (the first few get one extra row)."""

    size, extra = divmod(len(xs), parts)
    shards = []
    start = 0
    for part in range(parts):
        stop = start + size + (1 if part < extra else 0)
        shards.append((xs[start:stop], ys[start:stop]))
        start = stop
    return shards


def _worker(connection, model: TensorModule, loss_fn: LossFunction) -> None:
    """Worker loop: load weights, forward + backward on a slice, reply."""

    parameters = model.parameters()
    while True:
        message = connection.recv()
        if message is None:
            return
        weights, training, xs, ys = message
        for parameter, data in zip(parameters, weights):
            parameter.data = data
        model.train(training)
        loss = loss_fn(model, xs, ys)
        loss.backward()
        connection.send((loss.item(), [parameter.grad for parameter in parameters], target_count(ys)))
