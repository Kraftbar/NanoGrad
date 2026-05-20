# Local Tiny Shakespeare Data

Place `input.txt` here before running character language-model experiments on
Tiny Shakespeare.

Dataset text files are ignored by git.

Compare the tiny language-model configs on a capped slice:

```bash
python3 char_compare.py --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --output-dir /tmp/nanollm-language-run
```

Train the lightweight GPT-like character model:

```bash
python3 char_demo.py --preset lite-gpt --text-file data/tinyshakespeare/input.txt --max-chars 256 --validation-chars 64 --seed-text "First Ci"
```

See [../../docs/language_workflow.md](../../docs/language_workflow.md) for the
checkpoint evaluation and generation loop.
