# Local Multi30k Data

Multi30k task 1 (Elliott et al., 2016): English image descriptions from
Flickr30k with German translations. 29,000 training pairs, 1,014
validation pairs, and the 1,000-pair "test 2016" set.

```bash
for f in train.de train.en val.de val.en test_2016_flickr.de test_2016_flickr.en; do
  curl -L https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/$f.gz | gunzip > data/multi30k/$f
done
```
