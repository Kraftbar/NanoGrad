# The CUDA Backend

NanoGrad's kernels also run on an NVIDIA GPU. Each GPU file sits next to
its C twin:

| CPU | GPU |
| --- | --- |
| `kernels.c` | `kernels.cu` (same kernel names) |
| `kernels_c.py` | `kernels_cuda.py` |
| `test_kernels.py` | `test_kernels_cuda.py` |

Nothing above the kernels changes: `tensor.py`, the layers and the models
run as they are.

```bash
.venv/bin/pip install "nvidia-cuda-nvrtc>=13,<14"
NANOGRAD_BACKEND=cuda NANOGRAD_DTYPE=float32 .venv/bin/python diffusion_demo.py ...
```

or `kernels.use_backend("cuda")` before making any tensors. It needs only
the NVIDIA driver and that pip package; the first run compiles
`kernels.cu` (about 0.5 s) and caches the binary in `~/.cache/nanograd/`.

## How It Works

- **Compiling.** NVRTC compiles `kernels.cu` with `-DREAL=float` or
  `double`, the same trick as `kernels.c`. About a dozen driver calls
  through ctypes load and launch it.
- **Tensors stay on the GPU.** `Tensor.data` is a `DeviceArray`. Reading
  one from Python copies it to the host once; that copy is a *sync*,
  because Python waits for the GPU. An image model's training step has
  one sync, the loss it prints; the GPT's has three, with its token ids
  and gradient clipping (`kernels_cuda.syncs` counts them).
- **Launches don't wait.** Python builds the next op while the GPU runs
  the last one.
- **Memory.** Freed blocks go back to a pool instead of the driver, like
  PyTorch's caching allocator.

## Reading kernels.cu

The C loop `for (i = 0; i < n; i++)` becomes one GPU thread per `i`. The
sections go from easy to hard:

- **Elementwise:** one thread per value.
- **Rows** (softmax, layer norm): a warp shares values with
  `__shfl_xor_sync`.
- **Matrix products:** tiles in shared memory, double buffered; about
  8 TFLOP/s in float32.
- **Convolutions:** implicit GEMM, where tiles read patches through the
  index table, so no patch matrix exists in memory.
- **Scatters:** turned into gathers, so the sums run in a fixed order and
  give the same bits every run.
- **Recurrent** (LSTM, CRF): one launch walks all the time steps.

## Correctness

`test_kernels_cuda.py` checks every kernel against C (1e-12 in float64).
The whole suite also passes with `NANOGRAD_BACKEND=cuda`, apart from the
bit-identity and fork tests, which are skipped there.

## Speed

One training step in float32: RTX 3060 Ti against 8 threads of an idle
Ryzen 7 3700X (`gpu_benchmark.py`).

| model | GPU ms | CPU ms | gain |
| --- | ---: | ---: | ---: |
| LeNet-5 | 1.9 | 5.4 | 2.9x |
| ResNet, MNIST | 5.4 | 29.9 | 5.5x |
| ResNet-8, b32 | 9.5 | 82.6 | 8.7x |
| ResNet-8, b256 | 60.5 | 395 | 6.5x |
| ViT | 6.2 | 25.0 | 4.1x |
| baby GPT | 15.8 | 275 | 17x |
| GPT 6x384 | 222 | 12,176 | 55x |
| diffusion U-Net | 137 | 1,820 | 13x |

Small models are limited by Python, at about 6 µs per launch.

A whole diffusion run (MNIST, 20 epochs) takes 268 min on the CPU and
30 min on the GPU. The FIDs are 18.11 and 18.12, and the two runs print
the same losses.

## CUDA Graphs

A small model's step is about a thousand short kernels, and Python takes
longer to launch them than the GPU takes to run them. A CUDA graph
records one whole step (forward, backward, Adam) once; afterwards each
step copies the batch into fixed input arrays and replays every kernel
with one launch, as PyTorch's `torch.cuda.graphs` does:

```python
x = kernels.storage(pixels)                  # the graph's inputs
y = kernels_cuda.DeviceArray("l", labels)
step(x, y)                                   # one eager step first
graph = kernels_cuda.capture(lambda: step(x, y), x, y)
loss = graph(new_pixels, new_labels)         # step's loss tensor
```

`step` returns the loss tensor without reading it. `gpu_benchmark.py
--graph` does this for each model. A replay runs the same kernels on the
same addresses with the same arguments, so:

- **Memory.** The graph keeps every block its step uses.
- **No reads while recording** (nothing has run yet): `capture` raises.
  The loss is averaged on the GPU, and so is Adam's clipping factor.
- **Numbers that change every step** live in GPU memory. Adam's step
  size is worked out on the host before each replay, so learning-rate
  schedules still work.
- **State is updated in place** (weights, batch-norm statistics);
  `capture` raises if the step replaces an array it reads.
- **Python runs once**, while recording: a dropout seed or noise drawn
  there is fixed. The diffusion step draws its noise that way, so it
  cannot be recorded as it is.

Replays give the same bits as eager steps (`test_kernels_cuda.py`), where
the eager step itself repeats bit for bit; atomic adds (the conv image
gradient, repeated embedding ids) add in any order.

Measured while two other training runs kept the GPU busy (so both
columns are slower than the idle table above; medians of rounds that
alternate the two, and the gains vary by about 1.5x from run to run):

| model | eager ms | graph ms | gain |
| --- | ---: | ---: | ---: |
| LeNet-5 | 9.7 | 2.5 | 3.9x |
| ResNet, MNIST | 13.6 | 3.9 | 3.5x |
| ResNet-8, b32 | 19.1 | 14.8 | 1.3x |
| ViT | 13.3 | 4.5 | 2.9x |
| baby GPT | 29.4 | 11.8 | 2.5x |

Replays copy their inputs from pinned host memory without waiting: a
plain upload waits for the GPU, which takes milliseconds when another
program shares it.

## Limits

- 8 GB: the 6x384 GPT fits batch 32, but not batch 64.
- float64 runs 64x slower than float32 on a gaming GPU.
- CUDA does not survive `fork()`, so `parallel.py` stays on the CPU.
