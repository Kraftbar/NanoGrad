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
