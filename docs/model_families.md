# Model Families

Every model in NanoGrad sits on the same core: `tensor.py` (tensors and
autograd), the kernels, and the layers in `tensor_nn.py`. A new family
mostly *reuses* that stack; the interesting part is what little each one
had to add. Every family is checked against PyTorch running the same
architecture from the same weights (`torch_twin.py`, or PyTorch's own
`nn.LSTM`/`nn.GRU`): in float64 they train to the same numbers.

```text
                         lenet  resnet  vit  GPT  bert  lstm/gru  vae  diffusion
conv2d                     x      x                                        x
pad / stride                      x                                        x
batch norm                        x
avg / max pooling          x
upsample                                                                   x
linear                     x      x      x    x    x       x       x       x
attention                                x    x    x
layer / RMS norm                         x    x    x
embedding (row gather)                        x    x       x
row scatter                                   x
split / concat                           x    x    x       x               x
softmax + cross entropy    x      x      x    x    x       x
BCE with logits / MSE                                              x       x
```

(The GPT's row scatter is for its optional mixture-of-experts layers.)

## Classic CNN: LeNet (`vision.py`)

Two 5x5 convolutions with tanh and average pooling, then a linear layer:
LeNet-5 (1998) at 28x28. Convolution is im2col (gather every patch into a
row through a cached index table) followed by one matrix product.

*Added to the core*: convolution, pooling (index tables + C kernels).

## Residual networks: ResNet (`resnet.py`)

`out = relu(F(x) + shortcut(x))`: each block learns a correction to its
input, which is what lets very deep networks train. F is two 3x3
convolutions with batch norm; a stride-2 block halves the image and its
shortcut becomes a 1x1 convolution.

*Added*: padding (a scatter into a zero image) and stride for
convolution, batch norm (LayerNorm's math per channel, with running
statistics for evaluation), ReLU kernel.

## Vision transformer: ViT (`vit.py`)

Cut the image into 7x7 patches, embed each as a token, add learned
positions, run transformer blocks, average the tokens, classify.

*Added*: nothing but a `causal=False` switch on attention; the patching
is a reshape and a permute.

## GPT (`language.py`, `gpt_train.py`)

The character GPT follows nanoGPT; `--preset baby-llama` swaps in LLaMA's
parts and `--tokenizer bpe` uses byte-pair encoding. See
[llm_gears.md](./llm_gears.md) and [shakespeare_gpt.md](./shakespeare_gpt.md).

*Added*: embeddings, LayerNorm, GELU, softmax, attention (all heads in
one batched product), RMSNorm, rotary embeddings, SwiGLU, KV cache,
grouped-query and sliding-window attention, mixture-of-experts
feed-forward layers (`moe.py`, which needed a row-scatter op), LoRA
adapters (`lora.py`) and weight quantization (`quantize.py`).

## Encoders: BERT (`bert.py`, `bert_finetune.py`)

The GPT block without the causal mask, trained by masked character
modeling, then fine-tuned with a small classifier head to tell which
Shakespeare character is speaking (`speakers.py`): the pretrain/fine-tune
workflow behind GLUE results.

*Added*: masking recipe, mean-pooled classifier head.

## Recurrent networks: LSTM and GRU (`rnn.py`)

Read the sequence one step at a time, carrying a hidden state; gates
decide what to keep and what to write. Training unrolls the loop and
backpropagates through every step. `gpt_train.py --preset baby-lstm` is
Karpathy's char-rnn in the GPT's recipe.

`wikitext_lm.py` runs the same LSTM as a word-level language model on
WikiText-2 with the recipe of PyTorch's own `word_language_model`
example: the state carries over from one 35-word window to the next while
gradients stop at the window edge (truncated backpropagation through
time).

*Added*: `split` (whose pieces share one gradient buffer, so splitting a
sequence into T steps stays linear in T), sigmoid kernel, LSTM state in
and out, and a fused LSTM cell (two ops with C kernels instead of a
split and seven elementwise ops per step).

## Variational autoencoder (`vae.py`, `vae_compare.py`)

Encode an image to a distribution over codes, sample a code with the
reparameterization trick, decode, and train on reconstruction + KL
divergence. Decoding random codes generates new digits.

*Added*: binary cross entropy on logits (stable form, C kernel).

## Diffusion (`diffusion.py`, `diffusion_demo.py`)

Learn to predict the noise that was added to an image at a random noise
level; generate by starting from pure noise and removing it step by step.
The network is a small U-Net: convolutions that halve the image twice and
double it back, with skip connections and a sinusoidal time embedding.

*Added*: nearest-neighbour upsampling (an index-table gather), concat
through index tables.

## Graph networks: GCN (`gcn.py`)

Kipf and Welling's graph convolutional network on the Cora citation graph:
each layer averages every paper's features with those of the papers it
cites or is cited by, then applies a linear layer. Two layers classify
2,708 papers into 7 topics from the labels of 140. With pygcn's recipe it
reaches 83% test accuracy, trains identically to PyTorch (weights equal to
9e-13 after 200 steps), and runs 3x faster than PyTorch's sparse product
on this CPU.

*Added*: a sparse matrix (`SparseMatrix`, rows of nonzeros) and its
product with a dense matrix, a C kernel whose backward pass multiplies by
the stored transpose; Adam's L2 form of weight decay (`TensorAdam(l2=)`).
