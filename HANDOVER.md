# Handover: branch `perf-geometry-nanogpt-compare`

This is the state of NanoGrad when it left the cloud session (2026-10-08). The branch sits
on top of `origin/main` (`caef81f`) and holds about 220 commits that were never pushed. Across
164 files it adds about 29.5K lines and removes about 1.7K.

## Getting the work back into GitHub

The export zip holds three things:

- `NanoGrad/`: the full repository, with `.git`, every commit and the branch checked out.
- `nanograd-branch.bundle`: the same commits as a git bundle (`origin/main..branch`).
- `nanograd-branch.patch`: the same commits as one `git format-patch` file.

Pick the case that matches where you are:

- **From your existing clone**, fetch the branch out of the bundle and push it:

  ```bash
  git fetch /path/to/nanograd-branch.bundle perf-geometry-nanogpt-compare:perf-geometry-nanogpt-compare
  git push -u origin perf-geometry-nanogpt-compare
  ```

- **From the `NanoGrad/` folder in the zip**, its `origin` already points at the GitHub repo:

  ```bash
  cd NanoGrad && git push -u origin perf-geometry-nanogpt-compare
  ```

- **Without git history**, apply the patch: `git am nanograd-branch.patch`.

Then open a PR against `main` if you want one.

## Running it

Everything runs on Python 3 and a C compiler. The kernels build themselves into
`~/.cache/nanograd` on first import, with `-march=native`. Set `NANOGRAD_CFLAGS=""`
for a portable build.

Run the tests:

```bash
python3 -m unittest discover -p "test_*.py"      # about 700 tests; PyTorch comparisons are skipped
```

The PyTorch parity tests and every `--frameworks nanograd torch` comparison need PyTorch:

```bash
python3 -m venv .venv && .venv/bin/pip install torch
.venv/bin/python -m unittest discover -p "test_*.py"
```

Datasets are not in git (about 4.3 GB). Every `data/<name>/README.md` gives the download
command, and each family script's docstring gives its run command.

## Where things are

| what | where |
| --- | --- |
| Core | `tensor.py` (autograd), `kernels.c` (C kernels), `kernels_c.py` (ctypes), `kernels_reference.py` (readable Python versions), `kernels.py` (dispatch), `tensor_nn.py`, `losses.py`, `optim.py` |
| Model families | one file each: `recommend.py`, `uci.py`, `flow.py`, `sr.py`, `fewshot.py`, `adversarial.py`, `linkpred.py`, `audio.py`, `kws.py`, `distill.py`, `segment.py`, `tagger.py`, … each with its `test_*.py` |
| PyTorch twins | `torch_twin.py`, plus a `TorchRunner` in each family script |
| Results and speed tables | `docs/benchmarks.md` |
| Per-family write-ups | `docs/model_families.md` |
| LLM gears | `docs/llm_gears.md` (DPO, context extension, MoE, speculative decoding, …) |
| Speed suites | `benchmark.py` (op by op), `benchmark_suite.py` (models and families) |
| Logs | `runs/` (only the text logs are in the zip; checkpoints were left out) |

## State at handover

- **Speed** (one core, float64 unless noted):
  - Training steps take 0.6x to 1.4x as long as single-threaded PyTorch's.
  - The convolutional models are faster than PyTorch: 0.6x to 0.96x.
  - The slowest are the language models with a 33K-word output: WikiText LSTM 1.29x (float32 1.37x).
  - Newer families run at 0.7x to 1.4x.
- **Correctness**: every family trains to the same weights as its PyTorch twin, in float64
  and to rounding (1e-10 or better).
- **Latest additions**:
  - direct convolution for layers with at most 16 filters;
  - a vectorized binary cross entropy and softmax row max;
  - axis sums in C;
  - pad2d by whole rows;
  - a faster transposed-A gemm;
  - `logsumexp`;
  - the BiLSTM-CRF tagger (UD EWT test accuracy 94.66%; softmax output 94.45%).

## Open items

- **gemm tuning**: done. On the cloud Xeon (AVX-512), float64 reaches about 47 GFLOP/s against PyTorch's 62,
  and float32 95-110 against 125. On the Ryzen 7 3700X (AVX2) it is at or above PyTorch
  on 1 and 8 threads (docs/benchmarks.md); a sweep of the A block size found nothing better.
- **Large-vocabulary output layers**: these are the main remaining gap.
- **The tagger**: done. With the LSTM layers and the CRF as one op each, it runs at PyTorch's speed (25.5 vs 26.3 ms per step).
- **SLAM / geometry**: parked on purpose.

## Caveats

- **float32 twins drift**: in float32, NanoGrad and PyTorch drift apart over many steps
  (rounding amplified by unstable steps). The parity claims are about float64.
- **One relaxed test**: `test_fewshot` allows 1e-9 instead of 1e-10. C sums round
  slightly differently from Python's compensated `sum`, and three Adam steps amplify
  that difference.
- **Timing noise**: step timings vary by 10-20% between runs. The docs' tables were
  measured on an idle machine.
