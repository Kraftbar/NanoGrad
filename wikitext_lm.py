"""Word-level language modeling on WikiText-2, in NanoGrad and in PyTorch.

WikiText-2 (Merity et al., 2016) is a standard language-modeling benchmark:
2 million words of Wikipedia articles, 33,278 distinct words, scored by
perplexity on the held-out test articles (lower is better; perplexity P
means the model is as unsure as a fair choice among P words).

The recipe is PyTorch's own example (pytorch/examples, word_language_model):

- every line is split on spaces and ends with an ``<eos>`` token
- the text is cut into 20 long columns read in parallel, 35 words at a time;
  the LSTM state carries over from one window to the next, but gradients
  stop at the window edge (truncated backpropagation through time)
- a 2-layer LSTM, 200 wide, dropout 0.2; plain SGD at learning rate 20,
  gradients clipped to norm 0.25; the learning rate is divided by 4 after
  every epoch that does not improve validation loss

    python3 wikitext_lm.py --epochs 6                          # NanoGrad
    .venv/bin/python wikitext_lm.py --epochs 6 --framework torch
    .venv/bin/python wikitext_lm.py --compare --steps 50       # same weights, both frameworks
    python3 wikitext_lm.py --cell transformer --lr 5           # a 2-layer GPT instead of the LSTM

``--cell transformer`` is NanoGrad's GPT (language.py) at the example's
sizes: each 35-word window is read on its own, with no state carried
over, as in the example's Transformer model. Its PyTorch side is
torch_twin.py.

The data is in data/wikitext-2 ({train,valid,test}.txt, from the
pytorch/examples repository). Any folder of the same three files works:
``--data data/ptb --preset zaremba-small`` is the Penn Treebank and the
"small" LSTM of Zaremba et al. (2014), the classic word-level benchmark
(test perplexity 114.5 in the paper). Its recipe is the example's with
20-word windows, no dropout, every weight drawn from uniform(-0.1, 0.1),
and the learning rate halved after every epoch from the fifth on, for 13
epochs. (Zaremba sums the loss over the 20 steps at learning rate 1 with
gradients clipped to norm 5; on the mean loss used here that is learning
rate 20 and norm 0.25, the example's numbers.)
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from array import array
from pathlib import Path
from time import perf_counter

import kernels
from losses import linear_cross_entropy, softmax_cross_entropy
from optim import TensorSGD
from rnn import WordRNNModel, detach_state
from language import CharTransformerModel
from tensor import Tensor, no_grad
from train import evaluation_mode


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Corpus:
    """Word ids for the three splits; the vocabulary covers all three, in the
    order the example builds it, so ids (and so initial weights) match."""

    def __init__(self, folder: str | Path) -> None:
        self.words: list[str] = []
        self.ids: dict[str, int] = {}
        self.train, self.valid, self.test = (
            self._tokenize(Path(folder) / f"{split}.txt") for split in ("train", "valid", "test")
        )

    def _tokenize(self, path: Path) -> list[int]:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            for word in line.split() + ["<eos>"]:
                if word not in self.ids:
                    self.ids[word] = len(self.words)
                    self.words.append(word)
                out.append(self.ids[word])
        return out


def batchify(ids: list[int], batch_size: int) -> list[list[int]]:
    """Cut the text into ``batch_size`` equal columns (dropping the rest).
    Column b is read from start to end, 35 words per training step."""

    length = len(ids) // batch_size
    return [ids[b * length : (b + 1) * length] for b in range(batch_size)]


def windows(columns: list[list[int]], bptt: int):
    """(inputs, targets) for consecutive windows: targets are the inputs
    shifted by one word."""

    length = len(columns[0])
    for start in range(0, length - 1, bptt):
        size = min(bptt, length - 1 - start)
        yield ([column[start : start + size] for column in columns],
               [column[start + 1 : start + 1 + size] for column in columns])


# ---------------------------------------------------------------------------
# One runner per framework, same interface
# ---------------------------------------------------------------------------


class NanoGradRunner:
    def __init__(self, model: WordRNNModel, *, clip: float) -> None:
        self.model = model
        self.optimizer = TensorSGD(model.parameters(), lr=1.0, max_grad_norm=clip)
        self.state = None

    def reset(self) -> None:
        self.state = None

    def train_step(self, inputs, targets, lr: float) -> float:
        self.model.train()
        self.state = detach_state(self.state)
        self.optimizer.zero_grad()
        loss = self._loss(inputs, targets)
        loss.backward()
        self.optimizer.lr = lr
        self.optimizer.step()
        return loss.item()

    def eval_loss(self, inputs, targets) -> float:
        with no_grad(), evaluation_mode(self.model):
            return self._loss(inputs, targets).item()

    def _loss(self, inputs, targets):
        if isinstance(self.model, WordRNNModel):
            # The decoder and the loss in one op, a chunk of rows at a time:
            # the 33K-wide logits are never built whole (same results).
            hidden, self.state = self.model.features(inputs, self.state)
            decoder = self.model.decoder
            return linear_cross_entropy(hidden, decoder.weight, decoder.bias, [t for row in targets for t in row])
        logits = self.model.sequence_logits(Tensor.from_list(inputs))
        batch, length, vocab = logits.shape
        return softmax_cross_entropy(logits.reshape((batch * length, vocab)), [t for row in targets for t in row])

    def weights(self) -> list[list[float]]:
        return [list(parameter.data) for parameter in self.model.parameters()]

    def load_weights(self, weights) -> None:
        for parameter, values in zip(self.model.parameters(), weights):
            parameter.data = list(values)


class TorchRunner:
    """The example's RNNModel built from a NanoGrad model's weights
    (batch-first, so both frameworks see identical batches)."""

    def __init__(self, model: WordRNNModel, *, clip: float, dropout: float, dtype: str = "float64") -> None:
        import torch

        self.torch = torch
        self.clip = clip
        self.dtype = getattr(torch, dtype)
        cell = type(model.layers[0]).__name__                     # "LSTM" or "GRU"
        hidden = model.layers[0].hidden

        def p(tensor) -> "torch.Tensor":
            return torch.tensor(list(tensor.data), dtype=self.dtype).reshape(tensor.shape)

        self.encoder = torch.nn.Embedding(model.vocab_size, model.embedding.embedding_dim, dtype=self.dtype)
        self.rnn = getattr(torch.nn, cell)(model.embedding.embedding_dim, hidden, len(model.layers),
                                           dropout=dropout, batch_first=True, dtype=self.dtype)
        self.decoder = torch.nn.Linear(hidden, model.vocab_size, dtype=self.dtype)
        self.drop = torch.nn.Dropout(dropout)
        with torch.no_grad():
            self.encoder.weight.copy_(p(model.embedding.weight))
            for index, layer in enumerate(model.layers):
                getattr(self.rnn, f"weight_ih_l{index}").copy_(p(layer.input_weight))
                getattr(self.rnn, f"weight_hh_l{index}").copy_(p(layer.hidden_weight))
                if cell == "LSTM":
                    # PyTorch's LSTM has two bias vectors that are always
                    # added together; NanoGrad's has one. The second is held
                    # at zero, else SGD would move the sum twice as fast.
                    getattr(self.rnn, f"bias_ih_l{index}").copy_(p(layer.bias))
                    getattr(self.rnn, f"bias_hh_l{index}").zero_()
                    getattr(self.rnn, f"bias_hh_l{index}").requires_grad_(False)
                else:
                    getattr(self.rnn, f"bias_ih_l{index}").copy_(p(layer.input_bias))
                    getattr(self.rnn, f"bias_hh_l{index}").copy_(p(layer.hidden_bias))
            self.decoder.bias.copy_(p(model.decoder.bias))
            if model.tied:
                self.decoder.weight = self.encoder.weight
            else:
                self.decoder.weight.copy_(p(model.decoder.weight))
        self.modules = [self.encoder, self.rnn, self.decoder, self.drop]
        self.parameters = list(dict.fromkeys(
            parameter for module in self.modules for parameter in module.parameters() if parameter.requires_grad
        ))
        # The same weights in NanoGrad's order (an LSTM's two biases as one sum).
        self.layout = [[self.encoder.weight]]
        for index in range(len(model.layers)):
            weights = [getattr(self.rnn, f"{name}_l{index}") for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            self.layout += [[weights[0]], [weights[1]]] + ([weights[2:]] if cell == "LSTM" else [[weights[2]], [weights[3]]])
        self.layout += ([] if model.tied else [[self.decoder.weight]]) + [[self.decoder.bias]]
        self.state = None

    def reset(self) -> None:
        self.state = None

    def _loss(self, inputs, targets):
        torch = self.torch
        x = torch.tensor(inputs)
        output, self.state = self.rnn(self.drop(self.encoder(x)), self.state)
        logits = self.decoder(self.drop(output))
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(targets).reshape(-1))

    def train_step(self, inputs, targets, lr: float) -> float:
        torch = self.torch
        for module in self.modules:
            module.train()
        if self.state is not None:                                 # the example's repackage_hidden
            self.state = tuple(s.detach() for s in self.state) if isinstance(self.state, tuple) else self.state.detach()
        for parameter in self.parameters:
            parameter.grad = None
        loss = self._loss(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(parameter.grad, alpha=-lr)
        return loss.item()

    def eval_loss(self, inputs, targets) -> float:
        for module in self.modules:
            module.eval()
        with self.torch.no_grad():
            return self._loss(inputs, targets).item()

    def weights(self) -> list[list[float]]:
        return [sum(parts).detach().reshape(-1).tolist() for parts in self.layout]

    def load_weights(self, weights) -> None:
        with self.torch.no_grad():
            for parts, values in zip(self.layout, weights):
                parts[0].copy_(self.torch.tensor(values, dtype=self.dtype).reshape(parts[0].shape))
                for extra in parts[1:]:
                    extra.zero_()


class TorchTransformerRunner:
    """The GPT through torch_twin.py: PyTorch computes the same layers from
    the same weights (and draws its own dropout masks)."""

    def __init__(self, model: CharTransformerModel, *, clip: float, dtype: str = "float64") -> None:
        import torch

        from torch_twin import TorchTwin

        self.torch = torch
        self.clip = clip
        self.twin = TorchTwin(model, dtype=getattr(torch, dtype))
        self.parameters = self.twin.parameters

    def reset(self) -> None:
        pass

    def _loss(self, inputs, targets):
        logits = self.twin(self.torch.tensor(inputs))
        return self.torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), self.torch.tensor(targets).reshape(-1))

    def train_step(self, inputs, targets, lr: float) -> float:
        torch = self.torch
        self.twin.training = True
        for parameter in self.parameters:
            parameter.grad = None
        loss = self._loss(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(parameter.grad, alpha=-lr)
        return loss.item()

    def eval_loss(self, inputs, targets) -> float:
        self.twin.training = False
        with self.torch.no_grad():
            return self._loss(inputs, targets).item()

    def weights(self) -> list[list[float]]:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.parameters]

    def load_weights(self, weights) -> None:
        with self.torch.no_grad():
            for parameter, values in zip(self.parameters, weights):
                parameter.copy_(self.torch.tensor(values, dtype=parameter.dtype).reshape(parameter.shape))


def torch_runner(model, args: argparse.Namespace, *, dropout: float, dtype: str = "float64"):
    if isinstance(model, CharTransformerModel):
        return TorchTransformerRunner(model, clip=args.clip, dtype=dtype)
    return TorchRunner(model, clip=args.clip, dropout=dropout, dtype=dtype)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def evaluate(runner, columns: list[list[int]], bptt: int) -> float:
    """Mean loss per word (nats) over a whole split, reading it in order."""

    runner.reset()
    total, words = 0.0, 0
    for inputs, targets in windows(columns, bptt):
        total += runner.eval_loss(inputs, targets) * len(inputs[0])
        words += len(inputs[0])
    return total / words


def train_epoch(runner, columns, *, bptt: int, lr: float, epoch: int, log_every: int) -> None:
    runner.reset()
    batches = (len(columns[0]) - 1 + bptt - 1) // bptt
    recent, start = [], perf_counter()
    for index, (inputs, targets) in enumerate(windows(columns, bptt), start=1):
        recent.append(runner.train_step(inputs, targets, lr))
        if index % log_every == 0:
            loss = sum(recent) / len(recent)
            print(f"| epoch {epoch:3d} | {index:5d}/{batches:5d} batches | lr {lr:5.2f} | "
                  f"ms/batch {1000 * (perf_counter() - start) / len(recent):6.1f} | "
                  f"loss {loss:5.2f} | ppl {math.exp(loss):8.2f}", flush=True)
            recent, start = [], perf_counter()


def build_model(vocab_size: int, args: argparse.Namespace, *, dropout: float | None = None):
    model = _build_model(vocab_size, args, dropout=dropout)
    if args.init_range is not None:            # every weight from uniform(-r, r), as Zaremba et al.
        rng = random.Random(args.seed)
        for parameter in model.parameters():
            parameter.data = [rng.uniform(-args.init_range, args.init_range) for _ in range(parameter.numel)]
    return model


def _build_model(vocab_size: int, args: argparse.Namespace, *, dropout: float | None = None):
    if args.cell == "transformer":
        return CharTransformerModel(
            vocab_size, context_size=args.bptt, embedding_dim=args.emsize, hidden_dim=args.nhid,
            num_heads=args.nhead, num_layers=args.nlayers, feed_forward_activation="relu",
            tie_embeddings=args.tied, init="gpt2", seed=args.seed,
            dropout=args.dropout if dropout is None else dropout,
        )
    return WordRNNModel(
        vocab_size, embedding_dim=args.emsize, hidden=args.nhid, num_layers=args.nlayers, cell=args.cell,
        dropout=args.dropout if dropout is None else dropout, tied=args.tied, seed=args.seed,
    )


def run(args: argparse.Namespace) -> dict:
    corpus = Corpus(args.data)
    print(f"vocabulary {len(corpus.words)} words; train {len(corpus.train)}, "
          f"valid {len(corpus.valid)}, test {len(corpus.test)} tokens", flush=True)
    model = build_model(len(corpus.words), args)
    runner = (NanoGradRunner(model, clip=args.clip) if args.framework == "nanograd"
              else torch_runner(model, args, dropout=args.dropout, dtype=args.dtype or "float64"))
    train_columns = batchify(corpus.train, args.batch_size)
    valid_columns = batchify(corpus.valid, args.eval_batch_size)
    test_columns = batchify(corpus.test, args.eval_batch_size)

    lr, best, best_weights, history, first = args.lr, None, None, [], 1
    if args.checkpoint and Path(args.checkpoint).exists():       # resume after the last finished epoch
        state = load_checkpoint(args.checkpoint)
        runner.load_weights(state["weights"])
        lr, best, best_weights, history = state["lr"], state["best"], state["best_weights"], state["history"]
        first = len(history) + 1
        print(f"resuming after epoch {len(history)} from {args.checkpoint}", flush=True)
    for epoch in range(first, args.epochs + 1):
        start = perf_counter()
        train_epoch(runner, train_columns, bptt=args.bptt, lr=lr, epoch=epoch, log_every=args.log_every)
        valid = evaluate(runner, valid_columns, args.bptt)
        minutes = (perf_counter() - start) / 60
        print(f"| end of epoch {epoch:3d} | {minutes:5.1f} min | valid loss {valid:5.2f} | "
              f"valid ppl {math.exp(valid):8.2f}", flush=True)
        history.append({"epoch": epoch, "valid_ppl": math.exp(valid), "minutes": minutes, "lr": lr})
        if args.decay_after is not None:       # a fixed schedule (Zaremba et al.): test the last model
            best_weights = None
            if epoch >= args.decay_after:
                lr /= args.decay
        elif best is None or valid < best:
            best, best_weights = valid, runner.weights()
        else:
            lr /= 4.0                         # the example's annealing
        if args.checkpoint:
            save_checkpoint(args.checkpoint, weights=runner.weights(), lr=lr, best=best,
                            best_weights=best_weights, history=history)
    if best_weights is not None:
        runner.load_weights(best_weights)
    best = min(math.log(row["valid_ppl"]) for row in history) if best is None else best
    test = evaluate(runner, test_columns, args.bptt)
    print(f"| end of training | test loss {test:5.2f} | test ppl {math.exp(test):8.2f}", flush=True)
    return {"history": history, "valid_ppl": math.exp(best), "test_ppl": math.exp(test)}


def save_checkpoint(path: str, **state) -> None:
    """Everything needed to continue training after an epoch (weights as
    packed arrays, so 13M numbers take 100 MB rather than 300 MB of text).
    The dropout masks after a resume differ from an uninterrupted run's."""

    def pack(weights):
        return None if weights is None else [array("d", values) for values in weights]

    state["weights"], state["best_weights"] = pack(state["weights"]), pack(state["best_weights"])
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle)
    temporary.replace(path)                     # never leave a half-written checkpoint


def load_checkpoint(path: str) -> dict:
    with Path(path).open("rb") as handle:
        state = pickle.load(handle)
    for name in ("weights", "best_weights"):
        if state[name] is not None:
            state[name] = [list(values) for values in state[name]]
    return state


def compare(args: argparse.Namespace) -> dict:
    """Both frameworks from the same weights over the same windows (no dropout,
    so the math is identical): time per step and how far the weights drift."""

    import torch

    torch.set_num_threads(args.threads)
    corpus = Corpus(args.data)
    model = build_model(len(corpus.words), args, dropout=0.0)
    runners = {"torch": torch_runner(model, args, dropout=0.0, dtype=args.dtype or "float64"),  # copies the weights
               "nanograd": NanoGradRunner(model, clip=args.clip)}
    batches = list(windows(batchify(corpus.train, args.batch_size), args.bptt))[: args.steps]
    results = {}
    for name, runner in runners.items():
        start = perf_counter()
        losses = [runner.train_step(inputs, targets, args.lr) for inputs, targets in batches]
        results[name] = {"ms_per_step": 1000 * (perf_counter() - start) / len(batches), "losses": losses}
        print(f"{name:8s}: {results[name]['ms_per_step']:7.1f} ms/step, loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    difference = max(
        abs(a - b)
        for ours, theirs in zip(runners["nanograd"].weights(), runners["torch"].weights())
        for a, b in zip(ours, theirs)
    )
    print(f"largest weight difference after {len(batches)} steps: {difference:.2e}")
    results["max_weight_difference"] = difference
    return results


PRESETS = {
    # Zaremba, Sutskever and Vinyals (2014), "small" non-regularized LSTM on PTB.
    "zaremba-small": dict(bptt=20, dropout=0.0, epochs=13, decay_after=4, decay=2.0, init_range=0.1,
                          eval_batch_size=1),
    # Their "medium" regularized LSTM (test perplexity 82.7): 650 wide, dropout
    # 0.5, 35-word windows, 39 epochs, the rate divided by 1.2 after every epoch
    # from the seventh on. Their learning rate 1 and clip norm 5 on the loss
    # summed over 35 steps are 35 and 5 / 35 on the mean loss used here.
    "zaremba-medium": dict(emsize=650, nhid=650, bptt=35, dropout=0.5, epochs=39, lr=35.0, clip=5 / 35,
                           decay_after=6, decay=1.2, init_range=0.05, eval_batch_size=1),
    # And their "large" one (test perplexity 78.4): 1500 wide, dropout 0.65,
    # 55 epochs, the rate divided by 1.15 from the 15th, clip norm 10.
    "zaremba-large": dict(emsize=1500, nhid=1500, bptt=35, dropout=0.65, epochs=55, lr=35.0, clip=10 / 35,
                          decay_after=14, decay=1.15, init_range=0.04, eval_batch_size=1),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=tuple(PRESETS), help="another paper's recipe (other flags still override)")
    parser.add_argument("--data", default="data/wikitext-2")
    parser.add_argument("--framework", choices=("nanograd", "torch"), default="nanograd")
    parser.add_argument("--compare", action="store_true", help="time both frameworks on the same steps")
    parser.add_argument("--cell", choices=("lstm", "gru", "transformer"), default="lstm")
    parser.add_argument("--nhead", type=int, default=2, help="attention heads (transformer)")
    parser.add_argument("--emsize", type=int, default=200)
    parser.add_argument("--nhid", type=int, default=200)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=20.0)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--bptt", type=int, default=35)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--dtype", choices=("float64", "float32"))
    parser.add_argument("--steps", type=int, default=50, help="--compare: training steps")
    parser.add_argument("--threads", type=int, default=1, help="--compare / torch: PyTorch threads")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--decay-after", type=int, help="fixed schedule: divide the lr by --decay after this epoch")
    parser.add_argument("--decay", type=float, default=2.0)
    parser.add_argument("--init-range", type=float, help="draw every weight from uniform(-r, r)")
    parser.add_argument("--checkpoint", help="save after every epoch here, and resume from it if it exists")
    parser.add_argument("--seed", type=int, default=1111)
    preset = parser.parse_known_args(argv)[0].preset
    if preset:
        parser.set_defaults(**PRESETS[preset])
    args = parser.parse_args(argv)
    if args.dtype:
        kernels.set_default_dtype(args.dtype)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.compare:
        compare(args)
    else:
        if args.framework == "torch":
            import torch

            torch.set_num_threads(args.threads)
        run(args)


if __name__ == "__main__":
    main()
