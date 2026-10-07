# NanoGrad

NanoGrad is a small-scale machine learning research framework for exploring
automatic differentiation, tensor operations, small neural networks, and later
language and vision experiments from the ground up.

The project starts with a minimal autograd core and builds small, visible
learning checks on top of it before moving toward MNIST, LeNet-style vision,
and tiny language models.

## Architecture

NanoGrad is a stack of small layers. Each one only uses the layers below it,
and each can be read on its own:

```text
+---------------------------------------------------------------------+
| Recipes and benchmarks                                              |
| gpt_train | mnist_compare | bert_compare | nanogpt_compare | bench  |
+---------------------------------------------------------------------+
| Model families                                                      |
| lenet | resnet | vit | GPT | bert | rnn (LSTM, GRU) | vae           |
+---------------------------------------------------------------------+
| Layers, losses, optimizers, training                                |
| tensor_nn | losses | optim | schedules | train | parallel           |
+---------------------------------------------------------------------+
| Tensor and autograd: the math                                       |
| tensor.py: shapes, broadcasting, every op's forward + backward rule |
+---------------------------------------------------------------------+
| Kernels: the inner loops                                            |
| kernels_reference.py (Python, to read) <-> kernels.c (C, to run)    |
+---------------------------------------------------------------------+
```

The bottom split keeps the code readable without giving up speed:

- `tensor.py` says what an op computes and how its gradient flows.
- The kernels are the loops that do the heavy work (matrix products,
  im2col, pooling, softmax, LayerNorm, GELU, tanh, elementwise math, the
  Adam update). Each one is written twice: a short Python version to read
  (`kernels_reference.py`) and a C version to run (`kernels.c`).
  `kernels.py` picks one, and `test_kernels.py` checks that they agree.
- With the C kernels on, tensors keep their values in C arrays, so data
  goes from one kernel to the next without being copied through Python.

The model families share everything below them. A ResNet, a vision
transformer, a BERT-style encoder and an LSTM needed only a few new pieces
in the lower layers (padded and strided convolution, batch norm, ReLU,
sigmoid, split, a switch to turn off causal masking); the rest is reuse.
[docs/model_families.md](./docs/model_families.md) maps which core pieces
each family uses and what each one added. Each family is checked against
PyTorch running the same architecture from the same weights
(`torch_twin.py`): see [docs/benchmarks.md](./docs/benchmarks.md).

Measurements of where the time goes, and how far the stack is from PyTorch,
are in [docs/benchmarks.md](./docs/benchmarks.md).

## Milestones

### Current Foundation

- [x] Build a minimal scalar autograd engine
- [x] Add basic tensor operations
- [x] Implement simple neural network components from scratch
- [x] Add SGD and basic optimization tools
- [x] Validate gradients with small deterministic examples
- [x] Train scalar MLPs on line fitting, noisy line fitting, sign separator,
  tiny 2D clusters, AND, OR, and XOR checks
- [x] Add simple dataset and dataloader utilities
- [x] Track loss, accuracy, and basic runtime on small examples
- [x] Add a tensor binary MLP forward pass
- [x] Add tensor autograd for core tensor math
- [x] Add tensor module abstractions for linear layers and MLPs
- [x] Train tensor classifiers on AND, OR, and XOR logic gates
- [x] Add local MNIST IDX dataset loading
- [x] Add tensor multiclass loss, accuracy, and training helper
- [x] Add local-file MNIST MLP demo entry point
- [x] Add no-download channel-first MNIST-style CNN smoke demo
- [x] Add local-file MNIST CNN demo entry point
- [x] Add a two-conv LeNet-ish MNIST CNN preset
- [x] Add basic convolution and pooling tensor operations

### Next Small Checks

- [x] Report language-model perplexity and support checkpoint eval/generate-only loops
- [x] Compare capped Tiny Shakespeare train/validation curves across the bigram,
  embedding, and tiny GPT presets
- [x] Add a slightly larger `small-gpt` character-transformer preset
- [x] Export top-k/temperature character sampling grids for saved checkpoints
- [x] [Compare top-k sampled output against nanoGPT on the same capped text slice](./docs/language_nanogpt_comparison.md#side-by-side-runs-against-nanogpt)
- [x] Keep implementations readable rather than over-compressed
- [x] [Speed up the pure-Python core without hiding the math](./docs/performance.md)
- [x] Check every tensor op's backward rule against finite differences
- [x] GPT training building blocks: dropout, warmup + cosine learning rate,
  prompts shorter than the context, data-parallel training on all CPU cores
- [x] [Train nanoGPT's baby GPT on full Tiny Shakespeare](./docs/shakespeare_gpt.md)
  (val loss 1.92 vs nanoGPT's 1.89 with the same recipe)
- [x] Move matrix products, im2col, pooling, LayerNorm, GELU, softmax and
  tanh into optional C kernels (pure Python stays the reference)
- [x] [Benchmark against PyTorch: LeNet on MNIST and op by op](./docs/benchmarks.md)
- [x] [ResNet and ViT on MNIST, and a BERT-style encoder on Tiny Shakespeare,
  each trained identically to PyTorch](./docs/benchmarks.md#model-families)
- [x] [LLM gears: BPE tokenizer, KV cache, RMSNorm, rotary embeddings,
  SwiGLU](./docs/llm_gears.md)
- [x] Recurrent family: LSTM and GRU matching `torch.nn.LSTM`/`GRU`, and
  char-rnn in the same training recipe as the GPT (`--preset baby-lstm`)
- [x] Generative family: a VAE on MNIST, trained identically to PyTorch,
  and a diffusion model (DDPM, small U-Net) whose U-Net matches PyTorch
- [x] Pretrain-then-fine-tune: the BERT encoder learns speaker
  classification on Tiny Shakespeare (`bert_finetune.py`)
- [x] float32 tensors (`kernels.set_default_dtype("float32")`, `--dtype float32`)
- [x] Keep tensors in C arrays between kernels, so data stops being copied
  to and from Python lists at every op

### First Real Datasets

- [x] Run and tune MNIST MLP on local data
- [x] Train a LeNet-style MNIST model
- [x] Try a small CIFAR CNN after the MNIST path is stable
- [x] Fashion-MNIST and CIFAR-10 against PyTorch (`mnist_compare.py --dataset`)

### Later Branches

- [x] Explore a small AlexNet-inspired CIFAR model after convolution basics work
- [x] Train a tiny character-level language model
- [x] Add Tiny Shakespeare as the first language dataset
- [x] Add transformer/GPT-style components later
- [x] [Compare tiny language-model structure and behavior against nanoGPT](./docs/language_nanogpt_comparison.md)
- [x] [Document the character language-model experiment workflow](./docs/language_workflow.md)
- [x] Add safe character-model checkpoint evaluation and generation workflows
- [x] [Explore camera geometry and pose estimation](./docs/camera_geometry.md)
- [ ] Keep ORB-SLAM3 and DROID-SLAM as distant reference points
  ([how today's pieces map onto them](./docs/camera_geometry.md#how-this-maps-to-the-reference-points))
- [x] [Two-view bundle adjustment and RANSAC on the synthetic scene](./docs/camera_geometry.md#results-synthetic-matches-pose_demopy)
- [x] [Corner detection and matching on a rendered image pair](./docs/camera_geometry.md#results-rendered-image-pair-image_pair_demopy)
- [x] [Track points into a third view (PnP) to reject epipolar-consistent mismatches](./docs/camera_geometry.md#results-a-third-view-three_view_demopy)
- [ ] Chain frames into a small visual-odometry loop (PnP + sliding-window BA)
- [ ] Corner detection and matching on real camera images

NanoGrad does not aim to maximize benchmark scores by hiding complexity,
over-tuning parameters, or making the code unreadable. Benchmarks should help
guide design decisions, not replace them.

## Development Philosophy

NanoGrad is built in stages. The first goal is not to compete with PyTorch, nanoGPT, or SLAM systems directly. The first goal is to build the core ideas clearly and verify that they work.

The project favors:

- Small, readable implementations
- Correctness before performance
- Benchmarks on simple core components first
- Gradual expansion into LLM and SLAM experiments
- Reasonable LOC growth without forcing extreme code golf
- Keep the top-level README as a map, with detailed dataset and experiment
  notes living near the files they describe
- A **math-on-paper** code style: readable equations, matrices laid out as grids
  where practical, one conceptual operation per line, and explicit intermediate
  names over dense clever expressions

The project does **not** optimize only for lowest lines of code. Code should stay understandable, testable, and flexible enough to experiment with new ideas.

## Development Workflow

Use small, logical commits when they help preserve a clean checkpoint. Avoid
committing half-finished experiments or every tiny edit.

## Project Structure

```text
ref_micrograd/  # micrograd reference implementation
ref_nanogpt/    # nanoGPT reference implementation
ref_tinygrad/   # tinygrad reference implementation
data/            # datasets and dataloaders

datasets.py      # tiny synthetic datasets and batch helpers
engine.py        # scalar autograd core
tensor.py        # tensor container, tensor operations, and tensor autograd
gradcheck.py     # finite-difference gradient checks for tensor ops
tensor_nn.py     # tensor neural-network helpers (linear, layer norm, embeddings)
losses.py        # tensor loss functions
metrics.py       # accuracy and related metrics
nn.py            # scalar neural-network layers and activations
optim.py         # SGD, Adam, etc.
schedules.py     # learning-rate schedules (nanoGPT warmup + cosine)
parallel.py      # data-parallel gradients across worker processes
model.py         # scalar model definitions built from nn.py
train.py         # scalar and tensor training loops
demo.py          # scalar autograd training demo
tensor_demo.py   # tensor math and tensor-autograd demos
benchmark.py     # op-by-op timing against PyTorch (DeepBench-style)
mnist_compare.py # LeNet / ResNet / ViT on MNIST, Fashion-MNIST or CIFAR-10, NanoGrad vs PyTorch
bert_compare.py  # the BERT-style encoder on Tiny Shakespeare in both
vision.py        # reusable small CNN model implementations
resnet.py        # a small residual network (ResNet) with batch norm
vit.py           # a tiny vision transformer (ViT)
bert.py          # a tiny BERT-style encoder and its masking recipe
bpe.py           # byte-level byte-pair-encoding tokenizer (GPT-2 style)
rnn.py           # LSTM, GRU and a char-rnn language model
vae.py           # variational autoencoder (generative model)
vae_compare.py   # the VAE on MNIST in NanoGrad and PyTorch
diffusion.py     # denoising diffusion (DDPM) with a small U-Net
diffusion_demo.py # train it on MNIST and draw new digits (PNG)
speakers.py      # speaker classification task hidden in Tiny Shakespeare
bert_finetune.py # pretrain the BERT encoder, then fine-tune it on speakers
torch_twin.py    # runs a NanoGrad architecture in PyTorch with the same weights
mnist_demo.py    # local-file MNIST MLP demo
mnist_cnn_demo.py # local-file MNIST CNN demo
mnist_smoke_demo.py # no-download MNIST-style CNN smoke demo
cifar_cnn_demo.py # local-file CIFAR-10 CNN demo
text.py          # character text encoding and dataset helpers
language.py      # reusable small language model implementations
sampling.py      # greedy and top-k/temperature next-token sampling
char_checkpoint.py # character-model checkpoint save/load
char_demo.py     # tiny character-level language model demo
char_compare.py  # compare tiny character language-model configs
char_sample_grid.py # sweep checkpoint sampling settings into CSVs
nanogpt_compare.py # train NanoGrad and nanoGPT side by side (needs torch)
gpt_train.py     # nanoGPT-style step-based character GPT training
kernels.py       # compute kernels: picks the C or pure-Python backend
kernels_reference.py # the pure-Python kernels, written to be read
kernels.c        # the same kernels in C
kernels_c.py     # builds kernels.c and calls it through ctypes
language_benchmark.py # tiny transformer timing check
linalg.py        # small list-of-lists linear algebra (solve, eigen, SVD)
geometry.py      # pinhole camera, rotations, triangulation, PnP, essential matrix
ransac.py        # robust estimation: RANSAC for PnP and two-view geometry
bundle_adjustment.py # multi-view bundle adjustment (Schur complement, Huber)
features.py      # Harris corners, patch descriptors, matching, PGM files
pose_demo.py     # synthetic-scene camera pose estimation demo
image_pair_demo.py # rendered image pair -> corners -> matches -> pose
three_view_demo.py # third view: tracks, PnP, three-view bundle adjustment
```

## Reference Implementations

NanoGrad uses the following projects as reference points:

### Autograd / Tensor Engines
- **[micrograd](./ref_micrograd)**: Tiny scalar automatic differentiation baseline. (RedRef*)
- **[tinygrad](./ref_tinygrad)**: Minimal tensor and neural network framework baseline. (RedRef*)
- **PyTorch**: Production-grade tensor and autograd reference. External reference only.

### Language Modeling
- **[nanoGPT](./ref_nanogpt)**: Compact GPT-style language model training baseline. (RedRef*)

### SLAM / Robotics
- **ORB-SLAM3**: Classical visual SLAM reference.
- **DROID-SLAM**: Neural visual SLAM reference.

## Quick Checks

```bash
python3 -m unittest discover
python3 demo.py
python3 tensor_demo.py
python3 mnist_smoke_demo.py
python3 char_demo.py --preset tiny-transformer
python3 char_compare.py
python3 language_benchmark.py --repeat 1
python3 benchmark.py --repeat 1
python3 pose_demo.py --outliers 0.3
python3 image_pair_demo.py
python3 three_view_demo.py
```

The core has no third-party dependencies. Only the comparisons against
PyTorch need it:

```bash
python3 -m venv .venv && .venv/bin/pip install torch
.venv/bin/python nanogpt_compare.py --text-file data/tinyshakespeare/input.txt
.venv/bin/python mnist_compare.py --epochs 1 --model resnet
.venv/bin/python bert_compare.py --steps 500
.venv/bin/python benchmark.py --torch
```

Use the test suite as the main correctness check; `test_gradcheck.py` compares
every tensor op's backward rule against numerical gradients. The demos are small examples
for inspecting the scalar-autograd and tensor paths by eye.

For local MNIST and CIFAR-10 files, see [data/mnist](./data/mnist) and
[data/cifar10](./data/cifar10). The MNIST CNN demo can use
`python3 mnist_cnn_demo.py --preset lenet-ish`.

## Status

Early development.

---
*\*RedRef: Reference point for architectural comparison and performance benchmarking.*
