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
