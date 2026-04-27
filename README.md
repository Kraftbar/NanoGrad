# NanoGrad

NanoGrad is a small-scale machine learning research framework for exploring automatic differentiation, tensor operations, **self-supervised language modeling**, and **vision / SLAM-related learning components** from the ground up.

The project starts with a minimal autograd core and builds small experiments on top of it, including transformer-based language modeling and vision/SLAM experiments.

## Project Goals

- [ ] Build a minimal autograd and tensor engine
- [ ] Implement small neural network components from scratch
- [ ] Train tiny language models on public text datasets (self-supervised)
- [ ] Explore vision and SLAM-related learning tasks
- [ ] Benchmark against simple reference implementations

## Project Structure

```text
nanograd/
  core/        # tensors, autograd, operations
  nn/          # layers, losses, activations
  optim/       # SGD, Adam, etc.
  data/        # datasets and dataloaders
  train/       # training loops, checkpoints, logging
  llm/         # tokenizers, transformers, GPT-style models
  slam/        # camera models, pose estimation, mapping experiments
  benchmarks/  # speed, memory, loss, trajectory metrics
```

## Reference Implementations

NanoGrad uses the following projects as reference points:

### Autograd / Tensor Engines
- **[micrograd](./micrograd)**: Tiny scalar automatic differentiation baseline. (RedRef*)
- **[tinygrad](./tinygrad)**: Minimal tensor and neural network framework baseline. (RedRef*)
- **[pytorch](./pytorch)**: Production-grade tensor and autograd reference. (RedRef*)

### Language Modeling
- **[nanoGPT](./nanogpt)**: Compact GPT-style language model training baseline. (RedRef*)

### SLAM / Robotics
- **ORB-SLAM3**: Classical visual SLAM reference.
- **DROID-SLAM**: Neural visual SLAM reference.

## Datasets

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

## Status

Early development.

---
*\*RedRef: Reference point for architectural comparison and performance benchmarking.*
