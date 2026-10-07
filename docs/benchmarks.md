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

Time per training step (batch 32, forward + backward + Adam), full epoch:

| | ms / step | one epoch | vs PyTorch float64 |
| --- | ---: | ---: | ---: |
| PyTorch, float32, 4 threads | 5.1 | 10 s | 0.7x |
| PyTorch, float64, 1 thread | 7.4 | 14 s | 1x |
| NanoGrad, C kernels, 1 process | 17.8 | 33 s | 2.4x |
| NanoGrad, pure Python | 401 | 12.5 min | 55x |

So NanoGrad trains LeNet to 98% on MNIST in under two minutes on one core.
At the start of this work the C path took 176 ms per step; the history is
below. Four worker processes (`--workers 4`) no longer help here: LeNet is
now so small that sending weights and images to the workers costs as much
as the computation they save. They still pay off for the GPT.

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
| matmul 768x128 @ 128x512 | GPT | 56.1 | 2125 | 7.7 | 7.3x | 9e-14 |
| linear 768x128 -> 512 | GPT | 63.2 | 2284 | 7.1 | 9.0x | 9e-14 |
| layer_norm 768x128 | GPT | 3.6 | 56.4 | 0.80 | 4.6x | 2e-14 |
| gelu 768x512 | GPT | 16.9 | 116 | 4.0 | 4.2x | 2e-16 |
| softmax 12x4x64x64 | GPT | 6.8 | 42.8 | 1.4 | 5.0x | 4e-17 |
| cross_entropy 768x65 | GPT | 0.84 | 8.4 | 0.64 | 1.3x | 1e-19 |
| conv2d 32x1x28x28 * 6x1x5x5 | LeNet | 11.0 | 252 | 2.4 | 4.6x | 7e-13 |
| conv2d 32x6x12x12 * 16x6x5x5 | LeNet | 6.1 | 277 | 3.0 | 2.0x | 2e-13 |
| tanh 32x6x24x24 | LeNet | 6.2 | 17.3 | 0.71 | 8.7x | 4e-16 |
| avg_pool 2x2 32x6x24x24 | LeNet | 1.6 | 30.5 | 0.66 | 2.4x | 0 |
| max_pool 2x2 32x6x24x24 | LeNet | 1.9 | 19.6 | 1.1 | 1.8x | 0 |

## How The Gap Closed

The first measurements had NanoGrad's C path 9x behind PyTorch on LeNet,
and 6x to 35x behind op by op. Profiling showed two separate problems.

**Copying, not computing.** Tensors stored Python lists of floats, and C
needs packed arrays of doubles, so every kernel call copied its inputs into
arrays and its outputs back. That cost about 20 ns per number each way. For
tanh, about 70% of the time went to copying and about 15% to computing
tanh. The fix was one layer down, in storage: with the C kernels on, a
tensor's `data` and `grad` are C arrays (`array("d")`), so values go from
kernel to kernel with no copies. Python code still sees a flat sequence of
floats that indexes, slices and iterates like a list. With storage fixed,
the elementwise work moved into small kernels too: add, multiply, scale,
tanh, pooling, the cross-entropy softmax, and a fused Adam update.

**Matrix products: a plain C loop vs. a tuned BLAS.** On its own,
NanoGrad's C `gemm` runs at about 8 GFLOP/s, or about 13 with
`NANOGRAD_CFLAGS=-march=native`. PyTorch's BLAS reaches about 60 GFLOP/s on
one core of this machine. The kernel uses one simple trick: it computes
four rows of C per pass, so each loaded row of B is used four times.
Closing the rest of the gap needs packed panels and hand-written SIMD
micro-kernels: worth studying (see the BLIS papers), but not code for a
learning repo. Larger matrix products, as in the GPT, are now mostly gemm
time.

| LeNet step, C backend, 1 process | ms |
| --- | ---: |
| first measurement | 176 |
| tanh and pooling kernels, fewer copies around conv | 63 |
| tensors stored as C arrays | 27 |
| elementwise and loss kernels | 20 |
| fused Adam, cached index tables | 18 |

The same work took the baby GPT on Tiny Shakespeare from about 2 hours to
11 minutes ([shakespeare_gpt.md](./shakespeare_gpt.md)).

## Bigger Benchmarks

The standard lists (ImageNet/ResNet-50, COCO, GLUE/SQuAD, MLPerf) are built
from the same building blocks NanoGrad has, plus a few it does not yet have
(padded and strided convolution, batch norm, residual blocks). But they need
GPU-class compute: ResNet-50 is about 4 GFLOPs per image, on 1.2 million
images per epoch. At this scale the comparable steps are LeNet on MNIST
(above), a small CNN on CIFAR-10 (`cifar_cnn_demo.py` reads the binary
batches once they are placed in `data/cifar10/`), and the baby GPT on Tiny
Shakespeare ([shakespeare_gpt.md](./shakespeare_gpt.md)).
