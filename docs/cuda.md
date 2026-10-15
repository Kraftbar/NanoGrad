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

With the GPU otherwise idle (`gpu_benchmark.py --backends cuda --graph`):

| model | eager ms | graph ms | gain |
| --- | ---: | ---: | ---: |
| LeNet-5 | 1.5 | 0.7 | 2.1x |
| ResNet, MNIST | 4.6 | 2.8 | 1.6x |
| ResNet-8, b32 | 8.7 | 6.0 | 1.5x |
| ViT | 6.0 | 1.2 | 4.8x |
| baby GPT | 15.3 | 4.9 | 3.1x |

The gain is largest where Python's launches, not the GPU, set the pace.
While another program shares the GPU it grows (LeNet 3.9x), since each
eager launch then also waits its turn.

Replays copy their inputs from pinned host memory without waiting: a
plain upload waits for the GPU, which takes milliseconds when another
program shares it.

## Fused Kernels

`relu(x * w + b)` is three kernels, and each one reads its inputs from
GPU memory and writes its result back. They do one or two operations
per value, so that traffic is their whole cost. `fusion.py` is a small
compiler (a tiny `torch.compile`) that turns such chains into one
generated kernel, inside a CUDA graph:

```python
graph = fusion.capture(lambda: step(x, y), x, y)    # as kernels_cuda.capture
print(graph.program)                                 # what it fused
```

1. **Trace.** The step runs once; its launches, copies and clears are
   noted down instead of sent.
2. **Group.** Neighbouring elementwise kernels of one size become one:
   thread i computes value i of each in turn, in registers.
   `fusion.FUSIBLE` holds each kernel's arithmetic on one value, copied
   from kernels.cu: arithmetic, activations and their gradients, bias
   broadcasts, padding, gathers, copies, Adam.
3. **Drop stores.** A temporary that nothing reads later stays in a
   register.
4. **Batch.** One kernel on many independent arrays (Adam on each
   parameter, `zero_grad`) becomes one launch, which reads its pointers
   from a table, like PyTorch's multi-tensor apply.
5. **Compile.** NVRTC compiles the generated source once, cached on
   disk; pointers live in the table, so the source does not change.

A reduction, a matrix product or a convolution ends a group, and a
kernel may read other indexes than its own (a bias, a padded image)
only of arrays the group does not write. Only neighbours fuse, plus one
small piece moved out of the way (a bias's copy) when the two share no
array: then no reordering needs a proof.

**Bit for bit.** The compiler contracts `a * b + c` into one fused
multiply-add, rounded once. Across a former kernel boundary that would
change the result, so each value passed on is rounded explicitly
(`__fadd_rn(x, -0.0f)`, which cannot be contracted). Fused steps give
the same bits as unfused ones (`test_kernels_cuda.py`: every template in
both precisions, an MLP, a ViT, a GPT and batch norm over several
steps). Inside one kernel the compiler is free to contract or not, and
decided Adam's `w * decay - step` differently in a fused kernel, so
kernels.cu now rounds that product explicitly (`__fmul_rn`).

With the GPU otherwise idle (`gpu_benchmark.py --backends cuda --fuse
--rounds 5 --steps 10`: the columns timed one after another, five
times, best kept; the rounds agreed to a few percent). Nodes are a
step's kernels, copies and clears:

| model | graph ms | fused ms | nodes | fused |
| --- | ---: | ---: | ---: | ---: |
| LeNet-5 | 0.7 | 0.7 | 48 | 34 |
| ResNet, MNIST | 2.5 | 2.3 | 262 | 130 |
| ResNet-8, b32 | 6.0 | 5.7 | 262 | 130 |
| ResNet-8, b256 | 43.3 | 42.6 | 263 | 131 |
| ViT | 1.3 | 1.0 | 385 | 202 |
| baby GPT | 4.8 | 4.0 | 1214 | 674 |
| GPT 6x384 | 174 | 171 | 2202 | 1105 |
| diffusion U-Net | 82.2 | 75.7 | 831 | 444 |

Two different gains. Small models run about half the nodes, at roughly
1.5 µs each even inside a graph (ViT 1.3x, baby GPT 1.2x). The U-Net's
group norms, SiLUs and time embeddings are long elementwise chains over
big activations: one pass over memory instead of seven, 8% of its step.
Where matrix products and convolutions take the time (ResNet-8 at batch
256, the 6x384 GPT) fusion saves 1 to 2%, within the noise for the GPT.
Most of the remaining traffic is in kernels this does not touch: a bias
or activation fused into the product before it (an epilogue), and the
gradient-norm sums, two launches per parameter.

(The U-Net step is now recorded too: its noise and time steps come in
as graph inputs, drawn once by the benchmark for every column, so its
eager step is faster than in the first table, which drew them in Python
every step.)

## Limits

- 8 GB: the 6x384 GPT fits batch 32, but not batch 64.
- Two processes on one GPU: each pool keeps what it has freed, so cap
  them with `NANOGRAD_CUDA_MEMORY_MB`.
- float64 runs 64x slower than float32 on a gaming GPU.
- CUDA does not survive `fork()`, so `parallel.py` stays on the CPU.
- Fusion works only inside a CUDA graph; eager steps run kernel by kernel.
