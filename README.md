# NanoGrad

NanoGrad is a small-scale machine learning research framework for exploring automatic differentiation, tensor operations, **self-supervised language modeling**, and **vision / SLAM-related learning components** from the ground up.

The project starts with a minimal autograd core and builds small experiments on top of it, including transformer-based language modeling and vision/SLAM experiments.

## Project Goals

### Stage 1: Core Learning Engine

- [x] Build a minimal scalar autograd engine
- [x] Add basic tensor operations
- [x] Implement simple neural network components from scratch
- [x] Add SGD and basic optimization tools
- [x] Validate gradients with small deterministic examples

### Stage 2: Small Models

- [ ] Train tiny MLP-style models
- [ ] Add simple dataset and dataloader utilities
- [ ] Benchmark loss, speed, and memory on small examples
- [ ] Keep implementations readable rather than over-compressed

### Stage 3: Language Modeling

- [ ] Train a tiny character-level language model
- [ ] Add Tiny Shakespeare as the first language dataset
- [ ] Add transformer/GPT-style components later
- [ ] Compare structure and training behavior against nanoGPT

### Stage 4: Vision / SLAM Exploration

- [ ] Add basic vision utilities
- [ ] Explore camera geometry and pose estimation
- [ ] Add small visual odometry experiments
- [ ] Compare ideas against ORB-SLAM3 and DROID-SLAM as references

### Stage 5: Benchmarking

- [ ] Benchmark core operations
- [ ] Benchmark training loss and validation loss
- [ ] Track tokens per second where relevant
- [ ] Track memory usage
- [ ] Track SLAM trajectory error when SLAM experiments exist


Benchmarking starts with the basics. Early benchmarks should focus on whether the core engine works correctly and performs reasonably on small examples.

NanoGrad does not aim to maximize benchmark scores by hiding complexity, over-tuning parameters, or making the code unreadable. Benchmarks should help guide design decisions, not replace them.



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
data/           # datasets and dataloaders

engine.py    # scalar autograd core
tensor.py    # tensor container and non-autograd tensor operations
nn.py        # layers, losses, activations
optim.py     # SGD, Adam, etc.
model.py     # general model definitions built from nn.py
train.py     # training loops, checkpoints, logging
demo.py      # entry point for experiments and visualizations
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

## Datasets

### Synthetic / Toy

| Dataset / Check | Purpose | Status |
| --- | --- | --- |
| Line fitting | Scalar regression check for learning `y = ax + b` | ok |
| Sign separator | Binary classification check for separating negative and positive scalar inputs | ok |
| XOR logic gate | Non-linear binary classification check for hidden-layer MLPs | ok |
| AND / OR gates | Linear logic gate sanity checks | planned |
| Noisy line fitting | Regression check with imperfect data | planned |
| Tiny 2D clusters | Binary classification check with two input dimensions | planned |

Status key: `ok` means demo and test coverage exist; `partial` means the
check exists but is not fully verified; `planned` means not implemented yet.

### Language Modeling
- **Tiny Shakespeare**: Small smoke-test dataset.
- **TinyStories**: Small-model language learning dataset.
- **WikiText-2 / WikiText-103**: Clean language-modeling datasets.
- **OpenWebText**: GPT-2-style web text corpus.
- **Enwik8 / Enwik9**: Character-level Wikipedia benchmarks.

### Vision / SLAM
- **MNIST / Fashion-MNIST**: Basic model testing.
- **CIFAR-10 / CIFAR-100**: Small image benchmarks.
- **STL-10**: Unsupervised visual feature learning.
- **KITTI**: Autonomous driving / visual odometry benchmark.
- **TUM RGB-D**: RGB-D SLAM benchmark.
- **EuRoC MAV**: Visual-inertial SLAM benchmark.

## Benchmarking

NanoGrad benchmarks implementations using:

- Training loss
- Validation loss
- Tokens per second
- Memory usage
- Model size
- Dataset preprocessing time
- SLAM trajectory error
- Visual odometry drift

## Quick Checks

Run the scalar-autograd demo:

```bash
python3 demo.py
```

Run the current test suite:

```bash
python3 -m unittest discover
```

The current demos use tiny synthetic datasets as human-readable learning checks.

## Status

Early development.

---
*\*RedRef: Reference point for architectural comparison and performance benchmarking.*
