# The CUDA Backend

NanoGrad's kernels also run on an NVIDIA GPU. The GPU code sits next to its
C twin, so the two can be read side by side:

| CPU | GPU |
| --- | --- |
| `kernels.c`: the loops | `kernels.cu`: the same kernels (same names), as GPU threads |
| `kernels_c.py`: builds it, calls it through ctypes | `kernels_cuda.py`: compiles it with NVRTC, launches it through the driver |
| `test_kernels.py`: C against Python | `test_kernels_cuda.py`: CUDA against C |

Nothing above the kernels changes: `tensor.py`, the layers and every model
run as they are.

```bash
.venv/bin/pip install "nvidia-cuda-nvrtc>=13,<14"       # NVIDIA's runtime compiler, ~60 MB
NANOGRAD_BACKEND=cuda NANOGRAD_DTYPE=float32 .venv/bin/python diffusion_demo.py ...
```

or in code, before making any tensors:

```python
import kernels
kernels.use_backend("cuda")
kernels.set_default_dtype("float32")   # a gaming GPU runs float64 at 1/64 of float32's speed
```

It needs the NVIDIA driver (for `libcuda`) and the pip package above;
no CUDA toolkit, no build step. The first use compiles `kernels.cu` for
the GPU at hand, in about half a second, and caches the binary under
`~/.cache/nanograd/`.

## How It Works

**Compiling.** `kernels_cuda.py` hands the source of `kernels.cu` to
NVRTC with `-DREAL=float` or `-DREAL=double` (the same trick as
`kernels.c`), gets back a cubin for this GPU's architecture, and loads it
with `cuModuleLoadData`. Kernels are launched with `cuLaunchKernel`. That
is all the CUDA API there is: about a dozen driver calls through ctypes.

**Tensors live on the GPU.** With the CUDA backend, `Tensor.data` is a
`DeviceArray`: GPU memory that also behaves like a flat, read-only Python
sequence. Kernels take DeviceArrays and return new ones, so values stay
on the GPU from op to op. When Python code reads one (`x.data[i]`,
`sum(x.data)`, `loss.item()`), the whole array is copied to the host
once and kept until a kernel writes to it. Each such copy is a *sync*:
Python waits for the GPU to finish everything queued before it.
`kernels_cuda.syncs` counts them, which shows where code still touches
values from Python. Making the backend fast was mostly about removing
syncs: a training step now has three (the loss value, its gradient's
scale, and the loss that the training loop prints), four when clipping
gradients.

**Kernels are queued, not waited for.** A launch returns immediately, so
Python builds the next op while the GPU runs the last one. Only a sync
waits.

**Memory.** Freed DeviceArrays go back to a pool, kept by size, instead
of to the driver (`cuMemFree` waits for the GPU, and `cuMemAlloc` is
slow). Reusing a block without waiting is safe because every kernel runs
in order on one stream. This is PyTorch's caching allocator in small.

**Conventions.** A kernel takes only pointers, `double` (scalar settings,
as in `kernels.c`) and `long` (counts), so `launch()` packs any argument
list into one buffer without per-kernel declarations.

## Reading kernels.cu

The header of `kernels.cu` explains the one idea behind all of it: the C
loop `for (i = 0; i < count; i++) out[i] = ...` becomes one GPU thread
per `i`, and thousands run at once. From there each section shows one
technique, roughly in order of difficulty:

- **Elementwise** (`add_vectors`, `relu_forward`, ...): one thread per
  value. Neighbouring threads touch neighbouring memory, so a warp of 32
  threads loads one contiguous run in a single transaction.
- **Rows** (`softmax_rows`, `layer_norm_rows`): a row's maximum or sum
  needs threads to cooperate. A warp shares values with
  `__shfl_xor_sync`, so rows up to 1024 wide get one warp each and need
  no shared memory; wider rows get a block whose warps combine their
  totals in shared memory. Sums over all rows (a bias gradient) are done
  in two passes (column sums over chunks of rows, then the chunks added in
  order), so the result is the same every run.
- **Matrix products** (`gemm_*`): the classic tiled product. A block
  copies a 128 x 8 slice of A and an 8 x 128 slice of B into shared
  memory, and each of its 256 threads computes an 8 x 8 piece of the
  output from there, so every number loaded from GPU memory is used 128
  times. The next slices are loaded while the current ones are multiplied
  (double buffering). About 8 TFLOP/s in float32 at 1024^3, against the C
  gemm's 0.11 on one core.
- **Convolutions** (`conv_forward`, ...): implicit GEMM, the same tiled
  product, but its tile loads read image patches through the index table
  that `tensor.py` already builds, so no patch matrix ever exists in
  memory.
- **Scatters** (col2im, pooling and embedding gradients): many inputs add
  into the same output. With atomics the order of the additions, and so
  the last bits of a float32 result, changes from run to run. For the
  fixed index tables the scatter is turned around into a gather (each
  output thread adds up its own inputs, in the C loop's order), which
  gives the C kernel's bits exactly; atomics remain only where the
  indices exist only on the GPU (max-pool winners, embedding ids).
- **Recurrent** (`lstm_sequence`, `crf_forward`): time steps depend on
  each other. The LSTM launches two kernels per step (a small tiled
  product h W^T, then the cell); the CRF runs a block per sentence that
  walks all the words in one launch, with `__syncthreads()` between
  words.

## Correctness

`test_kernels_cuda.py` runs every kernel on both backends in float64 and
requires agreement to about 1e-12 (1e-5 in float32), on shapes chosen to
miss the tile sizes, and checks a few things only the GPU has (one sync
per array, the in-place `initial` of gemm). The whole test suite also runs
on the GPU:

```bash
NANOGRAD_BACKEND=cuda .venv/bin/python -m unittest discover
```

and passes, except for the tests that are skipped there: bit identity
with the fused C kernels (the GPU sums in a different order) and the
data-parallel tests that fork worker processes (CUDA does not survive a
fork).

## Speed

`gpu_benchmark.py`: one training step on random data, float32, RTX 3060 Ti
against the C kernels on 8 threads of a Ryzen 7 3700X (the CPU was also
running a training job, so its times are high by up to 2x):

| model | GPU ms | CPU ms | speedup |
| --- | ---: | ---: | ---: |
| LeNet-5, MNIST, batch 32 | 1.7 | 5.6 | 3.3x |
| small ResNet, MNIST, batch 32 | 5.0 | 34.4 | 6.9x |
| ResNet-8, CIFAR-10, batch 32 | 11.0 | 121.5 | 11x |
| ResNet-8, CIFAR-10, batch 256 | 67.9 | 559.8 | 8.2x |
| ViT, MNIST, batch 32 | 7.5 | 39.5 | 5.3x |
| baby GPT (4 x 128, context 64), batch 12 | 17.1 | 377.9 | 22x |
| GPT (6 x 384, context 256), batch 16 | 109.8 | 7,899 | 72x |
| diffusion U-Net (width 32, norm + attention), batch 128 | 144.5 | 2,566 | 18x |

The bigger the model, the bigger the gain. Small models are limited by
Python: a launch costs about 6 microseconds (3 of them in the driver),
and the baby GPT's step is about 1,000 launches plus the autograd
bookkeeping around them, while the GPU itself would need a few
milliseconds.

## A Whole Training Run

The MNIST diffusion model of [model_families.md](./model_families.md#diffusion-diffusionpy-diffusion_demopy)
(455K-parameter U-Net with group norm and attention, 20 epochs, EMA, DDIM
sampling), once on each backend, float32:

| | time | FID (same judge) |
| --- | ---: | ---: |
| C kernels, 8 threads | 268 min | 18.11 |
| CUDA | 30 min | 18.12 |

The two runs print the same loss, to the four digits shown, at every
report from step 100 to step 9,300: summing in a different order changes
the last bits of float32 values, not the training.

## Limits

- **Memory.** Every op's output stays alive until the backward pass, as on
  the CPU, but the GPU has only 8 GB: the 6 x 384 GPT needs about 220 MB
  per sequence of 256 tokens, so nanoGPT's batch of 64 does not fit
  (batch 16 does).
- **float64** runs at 1/64 of the float32 rate on a gaming GPU (about
  250 GFLOP/s here, still above the CPU), fine for checking, not for
  training.
- **One process.** CUDA does not survive `fork()`, so the data-parallel
  workers of `parallel.py` stay CPU-only.
