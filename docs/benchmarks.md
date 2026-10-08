# Benchmarks Against PyTorch

Every model family in NanoGrad is run on standard benchmarks next to
PyTorch. Each comparison works the same way: PyTorch builds the same
architecture (by hand, or with `torch_twin.py`, which walks the NanoGrad
model and runs each layer with PyTorch's functions), copies NanoGrad's
initial weights, and trains on the same batches in the same order. In
float64 the two then compute the same thing, so a comparison checks
correctness (the weights at the end agree to rounding) and measures
speed on identical work.

All runs are on one 4-core Intel Xeon at 2.1 GHz (AVX-512), PyTorch
2.14 for CPU. Unless a row says otherwise both frameworks use one core
and float64, and NanoGrad uses its C kernels.

## Results At A Glance

| benchmark | model | result (NanoGrad = PyTorch) | for reference |
| --- | --- | --- | --- |
| MNIST | LeNet-5, 5K parameters, 3 epochs | **98.00%** test accuracy | LeNet-5 paper (1998): 99.05%, 20 epochs |
| MNIST | small ResNet, 1 epoch | 97.84% | |
| CIFAR-10 | ResNet-8 (MLPerf Tiny's shape), 78K parameters, 3 epochs | **68.57%** | MLPerf Tiny target: 85% (after much longer training) |
| Fashion-MNIST / CIFAR-10 | small ResNet, 2 epochs | 86.90% / 58.30% | [all models below](#model-families) |
| Penn Treebank | Zaremba et al.'s small LSTM, 13 epochs | test perplexity **113.8** (PyTorch, same weights and batches: 115.0) | paper: test perplexity 114.5 |
| WikiText-2 | PyTorch's `word_language_model` LSTM, 6 epochs | test perplexity **135.4** (PyTorch, same weights and batches: 139.6; dropout masks differ) | |
| WikiText-2 | 2-layer GPT, same recipe | NanoGrad: running; PyTorch, same weights and batches: test perplexity 240.3 | |
| Cora (citation graph) | GCN, pygcn's recipe, 10 seeds | **83.1% +- 0.6** test accuracy (PyTorch, own dropout masks: 82.9% +- 0.7) | GCN paper: 81.5% (on the Planetoid split) |
| AG News (text classification) | fastText's model, words / words + bigrams, 5 epochs | **91.38% / 91.76%** | fastText paper: 91.5% / 92.5% |
| Cora (link prediction) | GAE / VGAE, the authors' recipe, 5 edge splits | test AUC **91.8 / 92.4**, AP 92.6 / 93.4 | GAE paper: AUC 91.0 / 91.4, AP 92.0 / 92.6 |
| Speech Commands (MLPerf Tiny keyword spotting) | DS-CNN, 12 classes, official test set, 24 epochs, no augmentation | **90.33%** test accuracy | MLPerf Tiny target: 90%; reference DS-CNN about 92% |
| MNIST (adversarial robustness) | fast FGSM adversarial training (Wong et al.), eps 0.3 | 97.95% clean, **75.81%** under PGD-50 x 10 | PyTorch, same recipe: 98.7% / 88.8% |
| CamVid (segmentation) | U-Net, half resolution, 20 epochs | 81.6% global accuracy, **36.9% mean IoU** | |
| MNIST (density) | MAF(5) normalizing flow, 20 epochs | test log-likelihood **-1217.6** nats (logit space), 1.74 bits/dim | MAF paper: -1300.5 |
| UCI regression (uncertainty) | MC dropout, Gal's protocol, 8 tables | beats the paper on 3 of 8 and is within a few percent on 4 (e.g. kin8nm RMSE 0.084, LL 1.07) | Gal and Ghahramani: kin8nm 0.10, 0.95 |
| MovieLens-1M (recommendation) | NCF: GMF / MLP / NeuMF, 32 factors, 20 epochs | HR@10 **0.710 / 0.691 / 0.695** | NCF code and reproductions: 0.708 / 0.692 / 0.701 |
| FSDD (spoken digits) | log-mel spectrogram + 3-block CNN, 20 epochs | **99.33%** test accuracy | |
| MNIST (distillation) | 2 x 800 student, alone / distilled from a 2 x 1,200 teacher, 20 epochs | **176 / 159** test errors (teacher 163) | Hinton et al.: 146 / 74 (teacher 67) |
| Cora | GAT (8 heads), early stopping, 5 seeds | 81.6% +- 1.6 | GAT paper: 83.0% +- 0.7 |
| ETTh1 (forecasting) | DLinear, 336 hours -> 96 / 192 / 336 / 720 | test MSE 0.370 / 0.421 / 0.455 / 0.494 | DLinear paper: 0.375 / 0.405 / 0.439 / 0.472 |
| Multi30k (German -> English) | encoder-decoder transformer, width 128, 2 + 2 layers, 10 epochs | test 2016 BLEU **32.5** greedy, **33.4** beam 4 | |
| Tiny Shakespeare | nanoGPT's baby GPT, 2,000 steps | val loss **1.921** | nanoGPT, same recipe: 1.886 |
| Tiny Shakespeare | BERT-style encoder, masked characters | identical in both | |
| MNIST | VAE, 3 epochs | **117.3** nats per image (negative ELBO) | |

The Tiny Shakespeare GPT is compared against nanoGPT itself (a separate
implementation, so the curves agree statistically, not to rounding); see
[shakespeare_gpt.md](./shakespeare_gpt.md) and
[llm_gears.md](./llm_gears.md) for its variants (LLaMA parts, BPE, LSTM,
mixture of experts).

## Model Families

```bash
.venv/bin/python mnist_compare.py --dataset cifar10 --model resnet --epochs 2
.venv/bin/python benchmark_suite.py        # every comparison, short, one table
```

`mnist_compare.py` trains LeNet-5 (two 5x5 convolutions with tanh and
average pooling), a small ResNet (3x3 convolutions, batch norm, residual
blocks, 20K parameters) and a tiny vision transformer (4x4 patches, 2
blocks, 28K parameters), all with Adam at lr 1e-3, batch 32, on the full
training sets; accuracy is on the full test sets.

| dataset | model | epochs | test accuracy | largest weight difference |
| --- | --- | ---: | ---: | ---: |
| MNIST | LeNet | 3 | 98.00% | 1e-15 |
| MNIST | ResNet | 1 | 97.84% | 8e-15 |
| MNIST | ViT | 1 | 92.60% | 9e-11 |
| Fashion-MNIST | LeNet | 2 | 83.22% | 9e-16 |
| Fashion-MNIST | ResNet | 2 | 86.90% | 1e-14 |
| Fashion-MNIST | ViT | 2 | 82.05% | 1e-10 |
| CIFAR-10 | LeNet | 2 | 44.09% | 3e-15 |
| CIFAR-10 | ResNet | 2 | 58.30% | 6e-15 |
| CIFAR-10 | ViT | 2 | 44.00% | 8e-11 |
| CIFAR-10 | ResNet-8 | 3 | 68.57% | 1.5e-12 |

"Largest weight difference" is between the NanoGrad and PyTorch weights
after the whole run: thousands of Adam steps later they still agree to
rounding error. (The vision transformer's 1e-10 is rounding too: its
LayerNorms and softmaxes amplify the last-bit differences between two
correct implementations a little more over 3,750 steps.)

These are short schedules on small models, chosen to finish in minutes
on a CPU, not to reach state-of-the-art accuracy.

## Speed

running

## How The Gap Closed

The first measurements had NanoGrad's C path 9x behind PyTorch on LeNet,
and 6x to 35x behind op by op. Three separate problems were in the way.

**Copying, not computing.** Tensors stored Python lists of floats, and C
needs packed arrays, so every kernel call copied its inputs into arrays
and its outputs back: about 20 ns per number each way. For tanh, about
70% of the time went to copying. The fix was one layer down, in storage:
with the C kernels on, a tensor's `data` and `grad` are C arrays
(`array("d")`, or `array("f")` in float32), so values go from kernel to
kernel with no copies. Python code still sees a flat sequence of floats.

**Python per element.** Elementwise math, losses, the Adam and SGD
updates, dropout masks, embedding lookups and the LSTM cell all moved into
small C kernels, each with a Python twin in `kernels_reference.py` that
the tests check it against.

**Matrix products.** A plain C loop runs a matrix product at about 8
GFLOP/s; PyTorch's BLAS reaches about 60 in float64 on one core of this CPU. `gemm` in `kernels.c` now
uses the ideas of every fast BLAS, in about 150 lines:

- a register tile: a 12 x 32 block of C (float32; 12 x 16 in float64)
  stays in 24 of AVX-512's 32 vector registers while the sum runs over k
  (6 rows with AVX2's 16 registers)
- packing: the columns of B a tile needs, and blocks of A, are copied
  into small contiguous buffers so the inner loop reads memory in order
- cache blocking: 256-deep slices of k and blocks of A of about 64K
  numbers, so each block is reused from cache by every column panel
- partial tiles at the edges with zero padding, instead of scalar loops

It reaches 40-47 GFLOP/s in float64 and 90-100 in float32 on one core.
Convolutions use the same micro-kernel without building the patch matrix
("implicit GEMM": the panel is packed straight from the image through an
index table), and softmax uses a vectorizable exp (range reduction plus a
Taylor polynomial, within 1 ulp of the C library's).

| LeNet step, batch 32, 1 process | ms |
| --- | ---: |
| first measurement | 176 |
| tanh and pooling kernels, fewer copies around conv | 63 |
| tensors stored as C arrays | 27 |
| elementwise and loss kernels | 20 |
| fused Adam, cached index tables | 18 |
| tiled gemm, implicit-GEMM convolution | 7.1 |
| PyTorch, float64, 1 thread | 6.7 |
