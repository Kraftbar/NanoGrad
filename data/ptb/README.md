# Local Penn Treebank Data

The Penn Treebank language-modeling set (Mikolov's preprocessing: 10,000
words, rare words as `<unk>`, numbers as `N`; about 930K training words)
is the benchmark of Zaremba, Sutskever and Vinyals (2014). Their copy:

```bash
for f in train valid test; do
  curl -L -o data/ptb/$f.txt https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.$f.txt
done
```

Then:

```bash
python3 wikitext_lm.py --data data/ptb --preset zaremba-small --dtype float32
```

Their "medium" LSTM, on the GPU (about 70 minutes; test perplexity 84.0):

```bash
NANOGRAD_BACKEND=cuda python3 wikitext_lm.py --data data/ptb --preset zaremba-medium --dtype float32
```

And the "large" one (about 5 hours; test perplexity 78.6): `--preset zaremba-large`.
