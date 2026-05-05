# Local MNIST Data

Place the standard MNIST IDX files in this directory before running
`mnist_demo.py`.

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

The demo does not download data automatically. This keeps tests and normal
demo runs deterministic and offline by default.
