# Local CIFAR-10 Data

Place CIFAR-10 binary batch files here before running CIFAR experiments.
The standard extracted `cifar-10-batches-bin/` directory also works.

Expected training files:

```text
data_batch_1.bin
data_batch_2.bin
data_batch_3.bin
data_batch_4.bin
data_batch_5.bin
```

Optional test file:

```text
test_batch.bin
```

Dataset binaries are ignored by git.

Check local files without training:

```bash
python3 cifar_cnn_demo.py --check-data
```

The official download is `cifar-10-binary.tar.gz` from
https://www.cs.toronto.edu/~kriz/cifar.html. Where that host is not
reachable, the NTU-ALComLab IWLS2021 repository keeps an unmodified copy of
the binary batches:

```bash
mkdir -p data/cifar10/cifar-10-batches-bin
for f in data_batch_1.bin data_batch_2.bin data_batch_3.bin data_batch_4.bin \
         data_batch_5.bin test_batch.bin; do
  curl -L -o data/cifar10/cifar-10-batches-bin/$f \
    https://raw.githubusercontent.com/NTU-ALComLab/IWLS2021/main/data/raw_bin/$f
done
```

Each file should be exactly 30,730,000 bytes (10,000 records of one label
byte plus 3,072 pixel bytes), and the test batch has 1,000 images per class.

Compare NanoGrad with PyTorch on it:

```bash
.venv/bin/python mnist_compare.py --dataset cifar10 --model resnet
```
