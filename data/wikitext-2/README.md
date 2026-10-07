# Local WikiText-2 Data

WikiText-2 (Merity et al., 2016) is about 2 million words of Wikipedia
articles that editors rated "good" or "featured", already tokenized into
words, with rare words replaced by `<unk>` (33,278 distinct words). It is a
standard word-level language-modeling benchmark, scored by perplexity on
the test articles.

`wikitext_lm.py` reads the copy that PyTorch's `word_language_model`
example ships with:

```bash
for f in train valid test; do
  curl -L -o data/wikitext-2/$f.txt \
    https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$f.txt
done
```

Then, for example:

```bash
python3 wikitext_lm.py --epochs 6 --dtype float32
```
