# Local UD English EWT Data

The English Web Treebank of Universal Dependencies (release 2.13,
CC BY-SA 4.0): 16.6K sentences of blogs, reviews, emails and answers,
with a universal part-of-speech tag (UPOS) on every word. `tagger.py`
reads the three CoNLL-U files:

```bash
for s in train dev test; do
  curl -L -o data/ud_ewt/en_ewt-ud-$s.conllu \
    https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.13/en_ewt-ud-$s.conllu
done
```
