# Local Cora Data

Cora (McCallum et al., 2000; LINQS) is a citation graph of 2,708
machine-learning papers in 7 topics, each described by 1,433 binary word
features, with 5,429 citation links. `gcn.py` reads the two files of
PyTorch's GCN reference implementation:

```bash
for f in cora.content cora.cites; do
  curl -L -o data/cora/$f https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/$f
done
```
