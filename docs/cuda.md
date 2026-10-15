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
- **Mixed precision:** the same products on the tensor cores, in PTX
  (see "Mixed Precision" below).
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
| LeNet-5 | 1.4 | 5.4 | 3.9x |
| ResNet, MNIST | 4.1 | 29.9 | 7.3x |
| ResNet-8, b32 | 7.6 | 82.6 | 11x |
| ResNet-8, b256 | 54.3 | 395 | 7.3x |
| ViT | 5.6 | 25.0 | 4.5x |
| baby GPT | 14.1 | 275 | 20x |
| GPT 6x384 | 146 | 12,176 | 83x |

Small models are limited by Python, at about 6 µs per launch; the three
sections below take that and more off.

A whole diffusion run (MNIST, 20 epochs) takes 268 min on the CPU and
30 min on the GPU. The FIDs are 18.11 and 18.12, and the two runs print
the same losses.

## CUDA Graphs

A small model's step is about a thousand short kernels. A CUDA graph
records one whole step (forward, backward, Adam) once, then replays it
with one launch, as `torch.cuda.graphs` does:

```python
step(x, y)                                   # one eager step first
graph = kernels_cuda.capture(lambda: step(x, y), x, y)
loss = graph(new_pixels, new_labels)         # copies the batch in, replays
```

A replay reuses the recorded addresses and arguments, so nothing may be
read back while recording, state is updated in place, and numbers that
change every step (Adam's step size) live in GPU memory. Anything Python
draws while recording (a dropout seed) is fixed. Replays give the same
bits as eager steps.

## Fused Kernels

`relu(x * w + b)` is three kernels, each reading its inputs from GPU
memory and writing its result back. `fusion.py` is a tiny `torch.compile`: it
traces a graph's kernels, merges neighbouring elementwise ones into one
generated kernel (values stay in registers), batches one kernel over
many arrays into one launch (Adam on every parameter), and compiles the
result with NVRTC. `fusion.capture` works like `kernels_cuda.capture`.
Each value passed between former kernels is rounded explicitly, so the
compiler can't fuse a multiply and an add across them, and the results
stay bit for bit the same.

## Mixed Precision

The tensor cores multiply 16-bit matrices about twice as fast as the
float32 kernels here (18 against 9 TFLOP/s). One switch puts every
matrix product on them:

```python
kernels.set_matmul_precision("bf16")   # or "fp16"; or NANOGRAD_AUTOCAST=bf16
```

The inputs are rounded to 16 bits as they are loaded and summed in
float32; weights, optimizer state, softmax and norms stay float32.
bfloat16 has float32's range and needs nothing else; float16 needs
`optim.GradScaler`. Baby GPT, 2,000 steps: val loss 1.8968 in float32,
1.8993 in bf16. On the CPU the switch rounds the same way, for testing.

## All Together

GPU ms per step, idle GPU (`--backends cuda --fuse --mixed bf16`):

| model | eager | graph | fused | + bf16 |
| --- | ---: | ---: | ---: | ---: |
| LeNet-5 | 1.4 | 0.7 | 0.7 | 0.6 |
| ResNet, MNIST | 4.1 | 2.5 | 2.3 | 1.9 |
| ResNet-8, b32 | 7.6 | 5.9 | 5.7 | 4.8 |
| ResNet-8, b256 | 54.3 | 42.7 | 42.8 | 35.5 |
| ViT | 5.6 | 1.3 | 1.0 | 1.1 |
| baby GPT | 14.1 | 4.5 | 3.7 | 3.1 |
| GPT 6x384 | 146 | 136 | 129 | 85 |
| diffusion U-Net | 78.6 | 79.3 | 72.4 | 48.3 |

Graphs help the small models (Python's launches), fusion the long
elementwise chains (the U-Net's norms), bf16 the big products. The
6x384 GPT ends at 85 ms, 143x the CPU.

## Limits

- 8 GB: the 6x384 GPT fits batch 32, not 64 (nothing is stored in 16
  bits, so mixed precision saves no memory).
- Two processes on one GPU: cap each pool with `NANOGRAD_CUDA_MEMORY_MB`.
- float64 runs 64x slower than float32 on a gaming GPU.
- CUDA does not survive `fork()`, so `parallel.py` stays on the CPU.
- Fusion only inside graphs; an fp16 step with `GradScaler` can't be
  recorded (it reads a value back).
