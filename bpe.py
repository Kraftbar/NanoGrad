"""Byte-level byte-pair encoding (BPE), the tokenizer family GPT-2 and later use.

A character model spends one prediction per character. BPE gives frequent
character sequences their own tokens: " the", "ing", "KING" can each become
one token, so a sequence of the same length covers more text.

Training, from raw UTF-8 bytes (256 starting tokens, so any text can be
encoded, never an "unknown character"):

1. Split the text into chunks with a regex: words with their leading
   space, numbers, punctuation runs, whitespace. Merges never cross a chunk
   boundary, so "dog." and "dog!" share the token for "dog".
2. Count every adjacent pair of tokens across all chunks (each unique chunk
   is counted once, weighted by how often it occurs, which keeps this fast).
3. The most frequent pair becomes a new token; replace it everywhere.
4. Repeat until the vocabulary has the requested size.

Encoding replays the merges in the order they were learned; decoding just
concatenates each token's bytes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


# GPT-2's split pattern, with ASCII letter and digit classes (Python's ``re``
# has no \p{L}). Every character of the input falls into some chunk.
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]]) -> None:
        """``merges[i]`` is the pair of tokens that token ``256 + i`` replaces."""

        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_bytes = [bytes([value]) for value in range(256)]
        for left, right in self.merges:
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("a byte-level vocabulary has at least 256 tokens")
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [(list(chunk.encode("utf-8")), count) for chunk, count in chunk_counts.items()]
        merges: list[tuple[int, int]] = []
        for new_token in range(256, vocab_size):
            pair_counts: Counter = Counter()
            for tokens, count in words:
                for pair in zip(tokens, tokens[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break                                  # nothing left to merge
            best = max(pair_counts, key=pair_counts.__getitem__)
            merges.append(best)
            words = [(_merge(tokens, best, new_token), count) for tokens, count in words]
        return cls(merges)

    def __len__(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            tokens.extend(self._cache[chunk])
        return tokens

    def _encode_chunk(self, chunk: str) -> list[int]:
        tokens = list(chunk.encode("utf-8"))
        while len(tokens) > 1:
            # Apply the earliest-learned merge that occurs in this chunk.
            pair = min(zip(tokens, tokens[1:]), key=lambda p: self.ranks.get(p, len(self.ranks)))
            if pair not in self.ranks:
                break
            tokens = _merge(tokens, pair, 256 + self.ranks[pair])
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        return b"".join(self.token_bytes[token] for token in tokens).decode("utf-8", errors="replace")

    def token_text(self, token: int) -> str:
        return self.token_bytes[token].decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"format": "nanograd.bpe.v1", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "nanograd.bpe.v1":
            raise ValueError("not a NanoGrad BPE tokenizer file")
        return cls([tuple(pair) for pair in payload["merges"]])


def _merge(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` (left to right)."""

    merged = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            merged.append(new_token)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged
