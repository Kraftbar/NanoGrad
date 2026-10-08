# Local ETT Data

The Electricity Transformer Temperature datasets (Zhou et al., Informer,
AAAI 2021): two years of hourly (ETTh) readings of oil temperature and six
load measurements from transformers in China. The long-horizon forecasting
benchmark of Informer, Autoformer, DLinear and PatchTST.

```bash
for f in ETTh1 ETTh2; do
  curl -L -o data/ett/$f.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f.csv
done
```
