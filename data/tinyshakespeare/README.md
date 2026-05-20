# Local Tiny Shakespeare Data

Place `input.txt` here before running character language-model experiments on
Tiny Shakespeare.

Dataset text files are ignored by git.

Compare the tiny language-model configs on a capped slice:

```bash
python3 char_compare.py --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --output-dir /tmp/nanollm-language-run
```

Train the GPT-like character model:

```bash
python3 char_demo.py --preset tiny-gpt --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --seed-text Firs
```

See [../../docs/language_workflow.md](../../docs/language_workflow.md) for the
checkpoint evaluation and generation loop.
