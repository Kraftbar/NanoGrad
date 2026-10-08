# Local Omniglot Data

Omniglot (Lake, Salakhutdinov and Tenenbaum, 2015): 1,623 handwritten
characters from 50 alphabets, 20 drawings each, as 105x105 1-bit PNGs.
The split files are Vinyals et al.'s (2016) few-shot split, as used by
the prototypical networks code: one line per class,
`Alphabet/characterNN/rotDDD`.

```bash
cd data/omniglot
for z in images_background images_evaluation; do
  curl -L -o $z.zip https://raw.githubusercontent.com/brendenlake/omniglot/master/python/$z.zip
done
for f in train val test trainval; do
  curl -L -o vinyals_$f.txt \
    https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/$f.txt
done
```

`fewshot.py` decodes the PNGs on first use (a few minutes) and caches the
28x28 images in `omniglot28.pkl`.
