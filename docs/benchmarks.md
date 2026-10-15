# Benchmarks Against PyTorch

Every model family in NanoGrad is run on standard benchmarks next to
PyTorch. Each comparison works the same way: PyTorch builds the same
architecture (by hand, or with `torch_twin.py`, which walks the NanoGrad
model and runs each layer with PyTorch's functions), copies NanoGrad's
initial weights, and trains on the same batches in the same order. In
float64 the two then compute the same thing, so a comparison checks
correctness (the weights at the end agree to rounding) and measures
speed on identical work.

All runs are on one 4-core Intel Xeon at 2.8 GHz (AVX-512), PyTorch
2.14 for CPU. Unless a row says otherwise both frameworks use one core
and float64, and NanoGrad uses its C kernels.

## Results At A Glance

| benchmark | model | result (NanoGrad = PyTorch) | for reference |
| --- | --- | --- | --- |
| MNIST | LeNet-5, 5K parameters, 3 epochs | **98.00%** test accuracy | LeNet-5 paper (1998): 99.05%, 20 epochs |
| MNIST | small ResNet, 1 epoch | 97.84% | |
| CIFAR-10 | ResNet-8 (MLPerf Tiny's shape), 78K parameters, 3 epochs; MLPerf's recipe (shifts and flips, L2 2e-4, lr x0.967 per epoch), float32, 8 threads | 68.57% after 3 epochs; **85.61%** after 150 epochs (best 85.95%, last ten 85.5-86.0%; 63 s per epoch, 792 MB steady) | MLPerf Tiny target: 85% |
| Fashion-MNIST / CIFAR-10 | small ResNet, 2 epochs | 86.90% / 58.30% | [all models below](#model-families) |
| Penn Treebank | Zaremba et al.'s small LSTM, 13 epochs | test perplexity **113.8** (PyTorch, same weights and batches: 115.0) | paper: test perplexity 114.5 |
| WikiText-2 | PyTorch's `word_language_model` LSTM, 6 epochs | test perplexity **135.4** (PyTorch, same weights and batches: 139.6; dropout masks differ) | |
| WikiText-2 | 2-layer GPT, same recipe | test perplexity **244.2** (PyTorch, same weights and batches: 240.3; dropout masks differ) | |
| Cora (citation graph) | GCN, pygcn's recipe, 10 seeds | **83.1% +- 0.6** test accuracy (PyTorch, own dropout masks: 82.9% +- 0.7) | GCN paper: 81.5% (on the Planetoid split) |
| AG News (text classification) | fastText's model, words / words + bigrams, 5 epochs | **91.38% / 91.76%** | fastText paper: 91.5% / 92.5% |
| Cora (link prediction) | GAE / VGAE, the authors' recipe, 5 edge splits | test AUC **91.8 / 92.4**, AP 92.6 / 93.4 | GAE paper: AUC 91.0 / 91.4, AP 92.0 / 92.6 |
| Speech Commands (MLPerf Tiny keyword spotting) | DS-CNN, 12 classes, official test set; the reference's dropout and L2, 36 epochs | **92.09%** test accuracy (without them, 24 epochs: 90.33%) | MLPerf Tiny target: 90%; reference DS-CNN about 92% |
| MNIST (adversarial robustness) | fast FGSM adversarial training (Wong et al.), eps 0.3 | 97.95% clean, **75.81%** under PGD-50 x 10 | PyTorch, same recipe: 98.7% / 88.8% |
| UD English EWT (part-of-speech tagging) | BiLSTM-CRF, word + suffix + shape embeddings (+ a character BiLSTM), 15 epochs | 94.66% test accuracy (softmax output instead of the CRF: 94.45%); with the character model **94.83%** (mean of 2 seeds 94.95%) | |
| CamVid (segmentation) | U-Net, half resolution, 20 epochs | 81.6% global accuracy, **36.9% mean IoU** | |
| MNIST (density) | MAF(5) normalizing flow, 20 epochs | test log-likelihood **-1217.6** nats (logit space), 1.74 bits/dim | MAF paper: -1300.5 |
| UCI regression (uncertainty) | MC dropout, Gal's protocol, 8 tables | beats the paper on 3 of 8 and is within a few percent on 4 (e.g. kin8nm RMSE 0.084, LL 1.07); yacht at Gal's 4,000 epochs: RMSE **0.705**, LL -1.257 | Gal and Ghahramani: kin8nm 0.10, 0.95; Gal's yacht results at 4,000 epochs: 0.666, -1.25 |
| MovieLens-1M (recommendation) | NCF: GMF / MLP / NeuMF, 32 factors, 20 epochs | HR@10 **0.710 / 0.691 / 0.695** | NCF code and reproductions: 0.708 / 0.692 / 0.701 |
| Omniglot (few-shot) | prototypical network, 5,000 episodes | 5-way 1-shot **98.3%**, 20-way 1-shot 94.4% | paper: 98.8%, 96.0% |
| Set5 / Set14 (super-resolution, x3) | SRCNN 9-1-5 on the 91-image set, 50 epochs | **32.11 / 28.80 dB** (bicubic 30.39 / 27.54) | SRCNN paper: 32.39 / 29.00 |
| FSDD (spoken digits) | log-mel spectrogram + 3-block CNN, 20 epochs | **99.33%** test accuracy | |
| MNIST (distillation) | 2 x 800 student, alone / distilled from a 2 x 1,200 teacher, 20 epochs | **176 / 159** test errors (teacher 163) | Hinton et al.: 146 / 74 (teacher 67) |
| Cora | GAT (8 heads), early stopping, 5 seeds | 81.6% +- 1.6 | GAT paper: 83.0% +- 0.7 |
| ETTh1 (forecasting) | DLinear, 336 hours -> 96 / 192 / 336 / 720 | test MSE 0.370 / 0.421 / 0.455 / 0.494 | DLinear paper: 0.375 / 0.405 / 0.439 / 0.472 |
| Multi30k (German -> English) | encoder-decoder transformer, width 128, 2 + 2 layers, 10 epochs | test 2016 BLEU **32.5** greedy, **33.4** beam 4 | |
| Tiny Shakespeare | nanoGPT's baby GPT, 2,000 steps | val loss **1.921** | nanoGPT, same recipe: 1.886 |
| Tiny Shakespeare | BERT-style encoder, masked characters | identical in both | |
| MNIST | VAE, 3 epochs | **117.3** nats per image (negative ELBO) | |
| MNIST (generation) | DDPM, 455K-parameter U-Net with group norm and 7 x 7 attention, 20 epochs, EMA, DDIM 100 steps | FID **18.1** on a small MNIST CNN's features, 1,000 samples (real images: 2.4; without norm and attention: 24.8) | |
| Fashion-MNIST (generation) | the same U-Net and recipe | FID **7.3** on a small Fashion-MNIST CNN's features, 1,000 samples (real images: 0.6) | |

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

Time per training step on the same work, both on one core, on an
otherwise idle machine (`mnist_compare.py`, `vae_compare.py`,
`bert_compare.py`, `wikitext_lm.py --compare`; logs in `runs/timing2/`). Every
row trains from the same weights on the same batches; in float64 the two
reach the same loss and accuracy.

| model | NanoGrad | PyTorch | NanoGrad / PyTorch |
| --- | ---: | ---: | ---: |
| LeNet, MNIST | 7.7 ms | 10.6 ms | 0.73x |
| LeNet, MNIST, float32 | 5.9 ms | 9.8 ms | 0.60x |
| LeNet, CIFAR-10 | 18.7 ms | 27.9 ms | 0.67x |
| small ResNet, MNIST | 58.4 ms | 68.9 ms | 0.85x |
| small ResNet, MNIST, float32 | 42.2 ms | 36.7 ms | 1.15x |
| small ResNet, CIFAR-10 | 84.8 ms | 104 ms | 0.82x |
| ResNet-8, CIFAR-10 | 188 ms | 197 ms | 0.96x |
| tiny ViT, MNIST | 15.8 ms | 18.0 ms | 0.88x |
| tiny ViT, CIFAR-10 | 23.3 ms | 24.9 ms | 0.94x |
| VAE, MNIST | 23.1 ms | 25.5 ms | 0.91x |
| BERT-style encoder, Tiny Shakespeare | 67.3 ms | 56.1 ms | 1.20x |
| WikiText-2 LSTM (2 x 200, 33K vocabulary) | 1,275 ms | 987 ms | 1.29x |
| WikiText-2 LSTM, float32 | 635 ms | 464 ms | 1.37x |

The convolutional models and the vision transformer run at PyTorch's
speed or faster in float64. The gap is in large matrix products: the
language models spend most of their step on the 200 -> 33,278 output
layer. On one core of this Xeon (2.8 GHz, AVX-512), NanoGrad's tiled
product reaches about 47 GFLOP/s in float64 and 95-110 in float32 on a
512^3 product; PyTorch's reaches about 62 and 125-135. (The two WikiText
rows were re-measured after tuning that output layer: a vectorized row max
for the 33,278-wide softmax, a one-pass cross-entropy gradient, and
packing a transposed A row by row for the weight gradient.) Op by op, forward plus backward
(`benchmark.py --torch`, float64, median of 20; these small ops vary by
10-20% between runs):

| op | NanoGrad C | PyTorch | ratio |
| --- | ---: | ---: | ---: |
| matmul 768x128 @ 128x512 | 8.32 ms | 6.30 ms | 1.3x |
| linear 768x128 -> 512 | 10.87 ms | 9.36 ms | 1.2x |
| layer norm 768x128 | 0.53 ms | 0.40 ms | 1.3x |
| GELU 768x512 | 4.75 ms | 4.52 ms | 1.1x |
| softmax 12x4x64x64 | 1.22 ms | 1.48 ms | 0.8x |
| cross entropy 768x65 | 0.73 ms | 0.57 ms | 1.3x |
| conv2d 32x6x12x12 * 16x6x5x5 | 2.05 ms | 1.80 ms | 1.1x |
| conv2d 3x3 pad 1, 32x8x28x28 * 8x8x3x3 | 12.65 ms | 15.36 ms | 0.8x |
| conv2d 3x3 stride 2 | 6.08 ms | 6.21 ms | 1.0x |
| batch norm 32x8x28x28 | 1.16 ms | 1.17 ms | 1.0x |
| ReLU 32x8x28x28 | 0.58 ms | 0.42 ms | 1.4x |
| max pool 2x2 | 0.28 ms | 0.99 ms | 0.3x |

The newer families, a few steps each in both frameworks
(`benchmark_suite.py --steps 30`; float64, one core; the last column is the largest
weight difference after those steps):

| comparison | NanoGrad ms/step | PyTorch ms/step | ratio | max weight difference |
| --- | ---: | ---: | ---: | ---: |
| NeuMF on MovieLens-1M | 11.1 | 12.8 | 0.9x | 5.3e-15 |
| VGAE on Cora | 220.9 | 277.1 | 0.8x | 4.5e-14 |
| MAF on MNIST | 136.1 | 105.1 | 1.3x | 6.8e-10 |
| SRCNN on 91-image | 112.3 | 110.9 | 1.0x | 1.4e-17 |
| prototypical net on Omniglot | 420.1 | 415.9 | 1.0x | 1.5e-10 |
| FGSM training on MNIST | 99.8 | 106.1 | 0.9x | 1.1e-13 |
| CNN on spoken digits | 193.9 | 176.0 | 1.1x | 2.4e-13 |
| DS-CNN keyword spotting | 295.9 | 406.7 | 0.7x | 7.8e-13 |
| BiLSTM-CRF tagger on UD EWT | 65.1 | 47.3 | 1.4x | 6.5e-13 |
| U-Net on CamVid | 2724.8 | 2541.0 | 1.1x | 1.2e-12 |

(The pure-Python reference path, kept to be read, is 30-300x slower than
the C kernels on the same ops.)

The tables above are from the cloud Xeon. On the Ryzen 7 3700X
(`benchmark_suite.py --steps 30`, 2026-10-15, machine otherwise idle),
NanoGrad is as fast as PyTorch or faster on every row except MAF (1.1x).
SRCNN was 1.7x until its 9 x 9 and 5 x 5 convolutions got their image
gradient as a full convolution with the flipped kernel instead of a
scatter (col2im):

| comparison | NanoGrad ms/step | PyTorch ms/step | ratio | max weight difference |
| --- | ---: | ---: | ---: | ---: |
| lenet on mnist | 5.0 | 6.7 | 0.7x | 6.9e-17 |
| resnet on mnist | 31.6 | 41.0 | 0.8x | 4.4e-13 |
| vit on mnist | 9.4 | 11.9 | 0.8x | 9.0e-12 |
| lenet on fashion | 4.8 | 7.5 | 0.6x | 1.1e-16 |
| resnet on fashion | 31.4 | 41.4 | 0.8x | 9.7e-15 |
| vit on fashion | 9.1 | 11.4 | 0.8x | 6.4e-12 |
| lenet on cifar10 | 11.2 | 20.7 | 0.5x | 3.3e-14 |
| resnet on cifar10 | 44.5 | 63.4 | 0.7x | 1.5e-15 |
| vit on cifar10 | 13.1 | 18.4 | 0.7x | 6.1e-12 |
| BERT masked characters | 41.0 | 41.0 | 1.0x | 4.3e-12 |
| VAE on mnist | 15.7 | 19.2 | 0.8x | 2.6e-15 |
| NeuMF on MovieLens-1M | 7.7 | 8.3 | 0.9x | 4.6e-15 |
| VGAE on Cora | 253.0 | 253.8 | 1.0x | 6.1e-13 |
| MAF on MNIST | 96.1 | 83.6 | 1.1x | 6.8e-10 |
| SRCNN on 91-image | 53.1 | 68.4 | 0.8x | 2.8e-17 |
| prototypical net on Omniglot | 303.9 | 346.0 | 0.9x | 1.2e-10 |
| FGSM training on MNIST | 65.2 | 84.0 | 0.8x | 4.3e-14 |
| CNN on spoken digits | 131.4 | 135.4 | 1.0x | 2.4e-13 |
| DS-CNN keyword spotting | 215.1 | 268.7 | 0.8x | 7.1e-13 |
| BiLSTM-CRF tagger on UD EWT | 26.9 | 27.3 | 1.0x | 5.9e-13 |
| U-Net on CamVid | 1675.9 | 1812.3 | 0.9x | 1.1e-12 |

WikiText-2 LSTM (`wikitext_lm.py --compare --steps 30`, one core, with
other programs running): float64 1,014 vs 1,374 ms per step (0.74x),
float32 578-619 vs 588-619 ms (about 1.0x), so the large-vocabulary gap
of the Xeon is gone here too. In float64 the weights agree to 5e-9 after
30 steps; in float32 rounding grows at lr 20 to a largest difference of
1.2 (the same with the old per-step LSTM ops), while the losses stay
close (7.93 vs 7.90).

The tagger's LSTM layers and its CRF now run as one op each
(`rnn.lstm_sequence`, `tagger.crf_forward`). With the old per-step ops it
took 41-46 ms against PyTorch's 26 ms on this machine (1.6-1.7x).

### Threads

`NANOGRAD_THREADS=8` (or `kernels.set_threads(8)`) lets the matrix
products, the convolutions and the row softmax use 8 threads. The default
is 1, so every table above is one core against one core. Threads only
split the output into blocks, so every number is summed in the same order
as on one core: results are the same bit for bit for any thread count
(`test_kernels.py` checks this). How it works is in
[performance.md](./performance.md#threads-inside-the-kernels).

These runs are on a different machine from the rest of this page: an
8-core AMD Ryzen 7 3700X (AVX2, no AVX-512), PyTorch 2.14 for CPU, with
another program keeping one core busy while they ran. Time per training
step, float64 unless noted, PyTorch with the same thread count
(`wikitext_lm.py --compare --steps 30 --threads N`,
`mnist_compare.py --torch-threads N`, MNIST, batch 32):

| model | NanoGrad, 1 thread | PyTorch, 1 thread | NanoGrad, 8 threads | PyTorch, 8 threads | ratio at 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiText-2 LSTM (2 x 200, 33K vocabulary) | 945 ms | 987 ms | 351 ms | 406 ms | 0.86x |
| WikiText-2 LSTM, float32 | 506 ms | 507 ms | 204 ms | 214 ms | 0.95x |
| small ResNet | 34.2 ms | 40.3 ms | 21.4 ms | 23.2 ms | 0.92x |
| LeNet | 4.9 ms | 6.6 ms | 4.1 ms | 4.7 ms | 0.87x |
| tiny ViT | 9.3 ms | 11.2 ms | 9.1 ms | 9.0 ms | 1.0x |

The WikiText rows use the fused output layer and loss
(`losses.linear_cross_entropy`): the 200 -> 33,278 decoder and the
softmax cross entropy are one op that never makes the softmax or its
gradient as separate 186 MB arrays, and works through the rows in chunks
when the logits would pass 256 MB. It computes exactly what `linear` then
`softmax_cross_entropy` did, bit for bit (same losses, same 4.75e-09
largest weight difference from PyTorch after 30 steps). Before it, the
same rows were 1,014 / 969 / 408 / 386 ms (float64) and 536 / 506 / 240 /
202 ms (float32): 1.06x and 1.19x PyTorch on 8 threads.

On their own the kernels scale 3-5x on 8 threads (the large-vocabulary
products 4-5x, the convolutions 4-7x). Whole steps gain less: the small
models' matrix products are too small to share (a block must be worth
about a million multiply-adds), and the rest of the step (elementwise
ops, Adam, the Python around them) still runs on one core.

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
On the Ryzen 7 3700X (Zen 2, AVX2, 2026-10-14), the same code is at or
above PyTorch 2.14's BLAS, GFLOP/s, best of 7:

| product (m x k x n) | float64, 1 thread | float32, 1 thread | float64, 8 threads | float32, 8 threads |
| --- | ---: | ---: | ---: | ---: |
| 1024^3 | 53 vs 51 | 114 vs 97 | 219 vs 180 | 535 vs 451 |
| 2048 x 512 x 512 | 54 vs 46 | 115 vs 99 | 247 vs 156 | 575 vs 441 |
| A^T, 512 x 2048 x 512 | 51 vs 47 | 112 vs 97 | 185 vs 151 | 466 vs 434 |

So here the gap is gone. Blocks of A from 16K to 128K numbers were also
tried: 16K and 32K are 5-10% slower, and 98K and 128K are within the
noise of 64K, so 64K stays.
Convolutions use the same micro-kernel without building the patch matrix
("implicit GEMM": the panel is packed straight from the image through an
index table), and softmax uses a vectorizable exp (range reduction plus a
Taylor polynomial, within 1 ulp of the C library's).

Large images needed two more changes (the U-Net step went from 3.5 s to
2.6 s, 1.1x PyTorch's). The index table is huge there, one entry per
(position, patch value), and reading it cost more than the arithmetic; but
each entry is a position's top-left pixel plus a fixed offset inside the
patch, so the kernels now read the position's base once and add the
offsets. And for stride 1 the image gradient is itself a convolution: the
output gradient, padded by the kernel size minus one, convolved with the
flipped kernel (channels and filters swapped), which runs on the fast
forward kernel instead of scattering patch by patch.

Layers with few filters needed the opposite of packing. With 8 filters a
register tile is 8 rows tall, and gathering each patch value into the
panel costs more than the 8 multiply-adds it feeds. `conv_direct` skips
patches altogether: a vector of output pixels along a row, for 8 filters
at once, accumulates `kernel[f, c, i, j] * image[c, y+i, x+j : x+j+V]`
over every (c, i, j), so one load feeds 8 multiply-adds and the 8 sums
never leave registers. It is used up to 16 filters (a float32 step of the small
ResNet went from 45 to 36 ms); the kernel gradient stays on the matrix
product, which was faster there.

| LeNet step, batch 32, 1 process | ms |
| --- | ---: |
| first measurement | 176 |
| tanh and pooling kernels, fewer copies around conv | 63 |
| tensors stored as C arrays | 27 |
| elementwise and loss kernels | 20 |
| fused Adam, cached index tables | 18 |
| tiled gemm, implicit-GEMM convolution | 7.1 |
| PyTorch, float64, 1 thread | 6.7 |
