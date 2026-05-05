# NanoGrad

NanoGrad is a small-scale machine learning research framework for exploring
automatic differentiation, tensor operations, small neural networks, and later
language and vision experiments from the ground up.

The project starts with a minimal autograd core and builds small, visible
learning checks on top of it before moving toward MNIST, LeNet-style vision,
and tiny language models.

## Architecture

NanoGrad is organized in two layers: a reusable core, then model and experiment
implementations built on top of that core.

```text
+-------------------------------------------------------------+
| Models / Experiments                                        |
| logic gates | MLPs | CNNs | tiny LMs | vision / SLAM        |
+-------------------------------------------------------------+
| Core                                                        |
| autograd | tensors | ops | losses | metrics | optimizers   |
+-------------------------------------------------------------+
```

Core modules should make many experiments possible. Model and experiment
modules should prove that the core works on increasingly realistic checks.

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

### Next Small Checks

- [ ] Train tensor MLPs on logic gates once the tensor training path exists
- [ ] Keep implementations readable rather than over-compressed

### First Real Datasets

- [ ] Train an MNIST MLP as the first image benchmark
- [ ] Add convolution and pooling tensor operations
- [ ] Train a LeNet-style MNIST model
- [ ] Try a small CIFAR CNN after the MNIST path is stable

### Later Branches

- [ ] Explore a small AlexNet-inspired CIFAR model after convolution basics work
- [ ] Train a tiny character-level language model
- [ ] Add Tiny Shakespeare as the first language dataset
- [ ] Add transformer/GPT-style components later
- [ ] Compare tiny language-model structure and behavior against nanoGPT
- [ ] Explore camera geometry and pose estimation
- [ ] Keep ORB-SLAM3 and DROID-SLAM as distant reference points

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
tensor.py        # tensor container and non-autograd tensor operations
tensor_nn.py     # non-autograd tensor neural-network helpers
losses.py        # tensor loss functions
metrics.py       # accuracy and related metrics
nn.py            # scalar neural-network layers and activations
optim.py         # SGD, Adam, etc.
model.py         # scalar model definitions built from nn.py
train.py         # scalar training loops
demo.py          # scalar autograd training demo
tensor_demo.py   # non-autograd tensor math demo
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
```

Use the test suite as the main correctness check. The demos are small examples
for inspecting the scalar-autograd and tensor paths by eye.

## Status

Early development.

---
*\*RedRef: Reference point for architectural comparison and performance benchmarking.*
