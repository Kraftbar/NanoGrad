# Language Experiment Log

This note records small local runs that affect language-model defaults. Keep
entries short and reproducible; detailed CSVs should live in run directories.

## 2026-05-20 Tiny Shakespeare 512-Char Slice

Common settings:

```bash
python3 char_compare.py --text-file data/tinyshakespeare/input.txt --max-chars 512 --validation-chars 128 --epochs 2 --batch-size 8 --sample-mode sample --temperature 0.8 --top-k 8 --num-samples 3 --generate 80
```

Summary:

| model | ctx | params | train loss | val loss | val ppl | val acc | dist-2 | rate | unit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bigram | 4 | 8145 | 3.0596 | 3.2130 | 24.85 | 0.202 | 0.522 | 107.5 | samples/s |
| embedding | 4 | 1877 | 1.4380 | 3.1071 | 22.36 | 0.355 | 0.695 | 303.4 | samples/s |
| tiny-gpt | 4 | 1053 | 2.9240 | 3.1009 | 22.22 | 0.200 | 0.602 | 209.9 | tokens/s |
| lite-gpt | 8 | 2189 | 2.4495 | 2.8948 | 18.08 | 0.214 | 0.705 | 134.8 | tokens/s |
| small-gpt | 8 | 7485 | 2.4825 | 2.9624 | 19.34 | 0.217 | 0.663 | 54.9 | tokens/s |

Notes:

- `lite-gpt` is the best transformer result in this short run and is much
  cheaper than `small-gpt`, so `char_compare.py` uses it as the default GPT-like
  comparison rung.
- `embedding` still learns the capped slice fastest and reaches the best
  validation accuracy. The transformer path needs more training and likely
  faster tensor kernels before it is a strong language model.
- Local artifact directories used for this entry:
  `/tmp/nanollm-language-run-20260520-2342`,
  `/tmp/nanollm-language-small-gpt-20260520-2343`, and
  `/tmp/nanollm-language-lite-default-20260520-2347`.
