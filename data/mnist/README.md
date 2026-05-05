# Local MNIST Data

Place the standard MNIST IDX files in this directory before running
`mnist_demo.py` or `mnist_cnn_demo.py`.

The official MNIST page is
[yann.lecun.org/exdb/mnist](https://yann.lecun.org/exdb/mnist/index.html).

Expected training files:

```text
data/mnist/train-images-idx3-ubyte.gz
data/mnist/train-labels-idx1-ubyte.gz
```

Plain IDX files also work:

```text
data/mnist/train-images-idx3-ubyte
data/mnist/train-labels-idx1-ubyte
```

Optional test files enable validation metrics:

```text
data/mnist/t10k-images-idx3-ubyte.gz
data/mnist/t10k-labels-idx1-ubyte.gz
```

The demo does not download data automatically. This keeps tests and normal
demo runs deterministic and offline by default.

Check local files without training:

```bash
python3 mnist_demo.py --check-data
python3 mnist_cnn_demo.py --check-data
```
