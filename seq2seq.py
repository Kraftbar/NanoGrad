"""The encoder-decoder transformer of "Attention Is All You Need" (2017).

A translation model reads a whole source sentence, then writes the
target sentence one token at a time:

    encoder:  source tokens -> bidirectional transformer blocks -> memory (S, C)
    decoder:  target so far -> blocks of
                  causal self-attention   (the target words written so far)
                  cross-attention         (queries from the target, keys and values from the memory)
                  feed-forward
              -> next-token scores

Everything is the GPT's and BERT's blocks again (language.py); the one new
piece is cross-attention, which is the same attention layer given a
``memory`` to attend over. Sentences in a batch have different lengths,
so they are padded, and a key mask hides the padding from attention.
Blocks are pre-norm, as in the GPT.
"""

from __future__ import annotations

from language import CausalSelfAttention, TokenPositionEmbedding, TransformerBlock, _feed_forward_activation
from tensor import Tensor
from tensor_nn import TensorDropout, TensorLayerNorm, TensorLinear, TensorModule


PAD, BOS, EOS = 0, 1, 2          # ids every vocabulary reserves


class DecoderBlock(TensorModule):
    """x += self_attention(norm1(x)); x += cross_attention(norm2(x), memory); x += ff(norm3(x))."""

    def __init__(self, width: int, context_size: int, *, heads: int, hidden: int, dropout: float = 0.0,
                 activation: str = "relu", seed: int = 0) -> None:
        self.activation = activation
        self.norm1 = TensorLayerNorm(width)
        self.self_attention = CausalSelfAttention(width, context_size, num_heads=heads, dropout=dropout, seed=seed)
        self.norm2 = TensorLayerNorm(width)
        self.cross_attention = CausalSelfAttention(width, context_size, num_heads=heads, dropout=dropout,
                                                   causal=False, seed=seed + 50)
        self.norm3 = TensorLayerNorm(width)
        self.feed_forward_in = TensorLinear(width, hidden, seed=seed + 100)
        self.feed_forward_out = TensorLinear(hidden, width, seed=seed + 101)
        self.dropout = TensorDropout(dropout, seed=seed + 102)

    def __call__(self, x: Tensor, memory: Tensor, memory_mask: Tensor | None) -> Tensor:
        x = x + self.self_attention(self.norm1(x))
        x = x + self.cross_attention(self.norm2(x), memory=memory, key_mask=memory_mask)
        hidden = _feed_forward_activation(self.feed_forward_in(self.norm3(x)), self.activation)
        return x + self.dropout(self.feed_forward_out(hidden))

    def parameters(self) -> list[Tensor]:
        parts = [self.norm1, self.self_attention, self.norm2, self.cross_attention, self.norm3,
                 self.feed_forward_in, self.feed_forward_out]
        return [parameter for part in parts for parameter in part.parameters()]


class Seq2SeqTransformer(TensorModule):
    def __init__(
        self, source_vocab: int, target_vocab: int, *, context_size: int = 64, width: int = 128,
        heads: int = 4, hidden: int = 256, encoder_layers: int = 2, decoder_layers: int = 2,
        dropout: float = 0.0, activation: str = "relu", seed: int = 0,
    ) -> None:
        self.source_embedding = TokenPositionEmbedding(source_vocab, context_size, width, seed=seed)
        self.target_embedding = TokenPositionEmbedding(target_vocab, context_size, width, seed=seed + 2)
        self.encoder = [
            TransformerBlock(width, context_size, hidden_dim=hidden, num_heads=heads, causal=False,
                             feed_forward_activation=activation, dropout=dropout, seed=seed + 10 + 10 * layer)
            for layer in range(encoder_layers)
        ]
        self.encoder_norm = TensorLayerNorm(width)
        self.decoder = [
            DecoderBlock(width, context_size, heads=heads, hidden=hidden, dropout=dropout, activation=activation,
                         seed=seed + 1000 + 1000 * layer)
            for layer in range(decoder_layers)
        ]
        self.decoder_norm = TensorLayerNorm(width)
        self.projection = TensorLinear(width, target_vocab, seed=seed + 5)
        self.embedding_dropout = TensorDropout(dropout, seed=seed + 6)

    def encode(self, source: list[list[int]]) -> tuple[Tensor, Tensor]:
        """The memory (B, S, C) and its key mask (B, S) for padded source ids."""

        mask = padding_mask(source)
        hidden = self.embedding_dropout(self.source_embedding(source))
        for block in self.encoder:
            hidden = block(hidden, key_mask=mask)
        return self.encoder_norm(hidden), mask

    def decode(self, target: list[list[int]], memory: Tensor, memory_mask: Tensor) -> Tensor:
        """Next-token scores (B, T, V) after each of the target tokens."""

        hidden = self.embedding_dropout(self.target_embedding(target))
        for block in self.decoder:
            hidden = block(hidden, memory, memory_mask)
        return self.projection(self.decoder_norm(hidden))

    def __call__(self, source: list[list[int]], target: list[list[int]]) -> Tensor:
        memory, mask = self.encode(source)
        return self.decode(target, memory, mask)

    def parameters(self) -> list[Tensor]:
        parts = [self.source_embedding, self.target_embedding, *self.encoder, self.encoder_norm,
                 *self.decoder, self.decoder_norm, self.projection]
        return [parameter for part in parts for parameter in part.parameters()]


def padding_mask(rows: list[list[int]]) -> Tensor:
    """(B, T): 0 where there is a token, -1e9 at padding."""

    width = len(rows[0])
    return Tensor([0.0 if token != PAD else -1e9 for row in rows for token in row], (len(rows), width))


def pad(rows: list[list[int]]) -> list[list[int]]:
    width = max(len(row) for row in rows)
    return [row + [PAD] * (width - len(row)) for row in rows]


def greedy_translate(model: Seq2SeqTransformer, source: list[list[int]], *, max_length: int) -> list[list[int]]:
    """Write each translation one token at a time, always taking the most
    likely next token, until every sentence has ended (EOS)."""

    from tensor import no_grad
    from train import evaluation_mode

    with no_grad(), evaluation_mode(model):
        memory, mask = model.encode(pad(source))
        outputs = [[BOS] for _ in source]
        done = [False] * len(source)
        for _ in range(max_length):
            scores = model.decode(outputs, memory, mask)                 # (B, T, V)
            length, vocab = scores.shape[1], scores.shape[2]
            for row in range(len(source)):
                if done[row]:
                    outputs[row].append(PAD)
                    continue
                start = (row * length + length - 1) * vocab
                last = scores.data[start : start + vocab]
                token = max(range(vocab), key=last.__getitem__)
                outputs[row].append(token)
                done[row] = token == EOS
            if all(done):
                break
    return [[token for token in row[1:] if token not in (PAD,)] for row in outputs]


def beam_translate(model: Seq2SeqTransformer, source: list[int], *, beam: int = 4, max_length: int,
                   length_penalty: float = 1.0) -> list[int]:
    """Beam search: keep the ``beam`` most likely partial translations at
    every step instead of only the single most likely next token.

    Every candidate is scored by its total log-probability divided by
    length ** ``length_penalty`` (otherwise short outputs win simply for
    having fewer probabilities multiplied together). A candidate that
    writes EOS is finished; the search stops when the best finished one
    beats every unfinished one.
    """

    import math

    from tensor import no_grad
    from train import evaluation_mode

    def score(log_probability: float, length: int) -> float:
        return log_probability / max(1, length) ** length_penalty

    with no_grad(), evaluation_mode(model):
        encoded = {}                                    # the source encoded once per row count
        alive = [([BOS], 0.0)]
        finished: list[tuple[list[int], float]] = []
        for _ in range(max_length):
            rows = len(alive)
            if rows not in encoded:
                encoded[rows] = model.encode([source] * rows)
            scores = model.decode([tokens for tokens, _ in alive], *encoded[rows])
            length, vocab = scores.shape[1], scores.shape[2]
            candidates = []
            for row, (tokens, total) in enumerate(alive):
                start = (row * length + length - 1) * vocab
                logits = scores.data[start : start + vocab]
                top = max(logits)
                log_sum = top + math.log(sum(math.exp(value - top) for value in logits))
                best = sorted(range(vocab), key=logits.__getitem__, reverse=True)[:beam]
                candidates += [(tokens + [token], total + logits[token] - log_sum) for token in best]
            candidates.sort(key=lambda c: score(c[1], len(c[0]) - 1), reverse=True)
            alive = []
            for tokens, total in candidates:
                if tokens[-1] == EOS:
                    finished.append((tokens, total))
                elif len(alive) < beam:
                    alive.append((tokens, total))
                if len(alive) == beam:
                    break
            best_finished = max((score(t, len(s) - 1) for s, t in finished), default=-math.inf)
            if not alive or best_finished >= max(score(t, len(s) - 1) for s, t in alive):
                break
        pool = finished or alive
        tokens, _ = max(pool, key=lambda c: score(c[1], len(c[0]) - 1))
    return [token for token in tokens[1:] if token not in (PAD, EOS)] + ([EOS] if tokens[-1] == EOS else [])
