# Benchmarks Against PyTorch

Two benchmarks compare NanoGrad with PyTorch:

1. a whole model, the classic LeNet CNN on MNIST
2. individual ops (matmul, conv, LayerNorm, ...) at the shapes real models
   use, like Baidu's DeepBench

Both check that the numbers are the same before timing anything. All runs
are on the same 4-core Intel Xeon (2.1 GHz). PyTorch is 2.14 (CPU).

## LeNet On MNIST

```bash
.venv/bin/python mnist_compare.py --epochs 3              # NanoGrad and PyTorch
python3 mnist_compare.py --frameworks nanograd --workers 4 # NanoGrad on 4 processes
```

The model is LeNet-5 scaled to 28x28 inputs: conv 6@5x5, tanh, 2x2 average
pool, conv 16@5x5, tanh, 2x2 average pool, linear 256 -> 10 (5,142
parameters). Training uses Adam (lr 1e-3), batch 32, and the full 60,000
training images. Test accuracy is on all 10,000 test images.

PyTorch starts from NanoGrad's initial weights and sees the same batches in
the same order. In float64 the two runs then compute the same thing:

| epoch | train loss | test accuracy | NanoGrad and PyTorch |
| ---: | ---: | ---: | --- |
| 1 | 0.3531 | 95.37% | identical |
| 2 | 0.1256 | 97.32% | identical |
| 3 | 0.0818 | **98.06%** | identical |

After all 5,625 Adam steps the largest weight difference between the two
is 2e-15. NanoGrad's pure-Python backend prints the same epoch-1 loss and
accuracy as well.

### Speed

Time per training step (batch 32, forward + backward + Adam):

| | ms / step | one epoch | vs PyTorch float64 |
| --- | ---: | ---: | ---: |
| PyTorch, float32, 4 threads | 5.1 | 10 s | 0.7x |
| PyTorch, float64, 1 thread | 7.3 | 14 s | 1x |
| NanoGrad, C kernels, 4 worker processes | 30 | 56 s | 4x |
| NanoGrad, C kernels, 1 process | 63 | 2 min | 9x |
| NanoGrad, pure Python | 401 | 12.5 min | 55x |

So NanoGrad trains LeNet to 98% on MNIST in about 6 minutes on one core, or
3 minutes on four. Before the tanh and pooling kernels and the
copy-avoidance changes, the C path took 176 ms per step.

## Op By Op

```bash
python3 benchmark.py                      # NanoGrad's C and Python backends
.venv/bin/python benchmark.py --torch     # plus PyTorch and an agreement check
```

Each row is one forward + backward pass with a fixed random output
gradient. Shapes come from the two models NanoGrad trains: the baby GPT
(batch 12 x context 64 = 768 rows, width 128, 4 heads) and LeNet (batch 32).
Times are medians of 5 runs, in milliseconds. PyTorch is float64 on one
thread. "max diff" is the largest difference from PyTorch in any output or
input gradient.

| op | model | NanoGrad C | NanoGrad Python | PyTorch | C / PyTorch | max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| matmul 768x128 @ 128x512 | GPT | 82.7 | 2034 | 7.0 | 12x | 9e-14 |
| linear 768x128 -> 512 | GPT | 111 | 2417 | 8.6 | 13x | 9e-14 |
| layer_norm 768x128 | GPT | 12.3 | 42.6 | 0.46 | 27x | 2e-14 |
| gelu 768x512 | GPT | 73.7 | 99.8 | 3.7 | 20x | 2e-16 |
| softmax 12x4x64x64 | GPT | 21.0 | 39.9 | 1.3 | 16x | 4e-17 |
| cross_entropy 768x65 | GPT | 6.6 | 6.4 | 0.51 | 13x | 1e-19 |
| conv2d 32x1x28x28 * 6x1x5x5 | LeNet | 12.4 | 208 | 2.0 | 6x | 7e-13 |
| conv2d 32x6x12x12 * 16x6x5x5 | LeNet | 7.5 | 200 | 1.6 | 5x | 2e-13 |
| tanh 32x6x24x24 | LeNet | 13.0 | 12.4 | 0.57 | 23x | 4e-16 |
| avg_pool 2x2 32x6x24x24 | LeNet | 4.7 | 19.0 | 0.60 | 8x | 0 |
| max_pool 2x2 32x6x24x24 | LeNet | 5.5 | 16.0 | 0.99 | 6x | 0 |

## Where The Remaining Time Goes

Two different gaps, with two different fixes.

**Matrix products: the C loop vs. a tuned BLAS.** On its own, NanoGrad's C
`gemm` runs at about 8 GFLOP/s, or about 13 with
`NANOGRAD_CFLAGS=-march=native`. PyTorch's BLAS reaches about 60 GFLOP/s on
one core of this machine. Closing that gap needs packed panels and
hand-written SIMD micro-kernels: worth studying (see the BLIS papers), but
not code to put in a learning repo. The kernel uses the simplest trick that
helps: four rows of C per pass, so each loaded row of B is used four
times.

**Everything else: copying, not computing.** Tensors store their values in
Python lists of floats, and C needs packed arrays of doubles. Every kernel
call converts its inputs from lists to arrays and its outputs back. Each
conversion costs about 20 ns per number, and so does the Python-side
bookkeeping for each float object. For tanh, a profile shows about 70% of
the step converting data and about 15% computing tanh. That is why C tanh
is no faster than Python tanh, which pays roughly the same per number to
call `math.tanh`.

The fix for the second gap is one layer lower: keep tensor data in C arrays
from one kernel to the next, and convert only at the edges (creating a
tensor from Python values, or reading `.data`). `conv2d` already does this
inside one op: its im2col patch matrix is gathered, multiplied and reused
in the backward pass without ever becoming a Python list. That is part of
why conv is the closest op to PyTorch. Doing the same for every tensor is
the next milestone in the README.

## Bigger Benchmarks

The standard lists (ImageNet/ResNet-50, COCO, GLUE/SQuAD, MLPerf) are built
from the same building blocks NanoGrad has, plus a few it does not yet have
(padded and strided convolution, batch norm, residual blocks). But they need
GPU-class compute: ResNet-50 is about 4 GFLOPs per image, on 1.2 million
images per epoch. At this scale the comparable steps are LeNet on MNIST
(above), a small CNN on CIFAR-10 (`cifar_cnn_demo.py` reads the binary
batches once they are placed in `data/cifar10/`), and the baby GPT on Tiny
Shakespeare ([shakespeare_gpt.md](./shakespeare_gpt.md)).
