# Local Fashion-MNIST Data

Fashion-MNIST (Zalando Research) is a drop-in replacement for MNIST: the same
file format and sizes (60,000 training and 10,000 test images, 28x28
grayscale, 10 classes), but of clothing (T-shirt, trouser, pullover, dress,
coat, sandal, shirt, sneaker, bag, ankle boot) instead of digits. It is
noticeably harder, so it separates models better.

Download the four files from the official repository into this directory:

```bash
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -L -o data/fashion_mnist/$f \
    https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f
done
```

Then run any MNIST script on it, for example:

```bash
.venv/bin/python mnist_compare.py --data-dir data/fashion_mnist --model resnet
```
