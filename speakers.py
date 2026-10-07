"""A labeled text task hidden in Tiny Shakespeare: who is speaking?

Every speech in the text starts with the speaker's name on its own line:

    ROMEO:
    But, soft! what light through yonder window breaks?

That turns the text into a classification dataset for free: given a window
of a speech, predict which of the most talkative characters said it. It
stands in for benchmarks like GLUE, where a pretrained language model is
fine-tuned to label sentences.

Each speaker's speeches are split in order: the first 90% for training,
the last 10% held out. ``pretraining_text`` is the whole text with the
held-out speeches removed, so a model pretrained on it never saw them.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

SPEECH = re.compile(r"(?m)^([A-Z][A-Za-z ]+):\n((?:.+\n)+)")


@dataclass
class SpeakerData:
    speakers: list[str]                       # class index -> name
    train: list[tuple[str, int]]              # (speech text, class)
    validation: list[tuple[str, int]]
    pretraining_text: str


def speaker_data(text: str, *, num_speakers: int = 8, validation_fraction: float = 0.1) -> SpeakerData:
    speeches = [(match.group(1), match.group(2), match.span(2)) for match in SPEECH.finditer(text)]
    volume = Counter()
    for name, body, _ in speeches:
        volume[name] += len(body)
    speakers = [name for name, _ in volume.most_common(num_speakers)]
    label = {name: index for index, name in enumerate(speakers)}

    train, validation, held_out_spans = [], [], []
    for name in speakers:
        own = [(body, span) for speaker, body, span in speeches if speaker == name]
        cut = int(len(own) * (1.0 - validation_fraction))
        train += [(body, label[name]) for body, _ in own[:cut]]
        validation += [(body, label[name]) for body, _ in own[cut:]]
        held_out_spans += [span for _, span in own[cut:]]

    pieces, position = [], 0
    for start, end in sorted(held_out_spans):
        pieces.append(text[position:start])
        position = end
    pieces.append(text[position:])
    return SpeakerData(speakers, train, validation, "".join(pieces))


def speech_windows(
    examples: list[tuple[str, int]], *, length: int, count: int, rng: random.Random,
) -> tuple[list[str], list[int]]:
    """``count`` random examples: a ``length``-character window of a speech
    (padded with spaces when the speech is shorter) and its speaker."""

    texts, labels = [], []
    for _ in range(count):
        body, speaker = examples[rng.randrange(len(examples))]
        start = rng.randrange(max(1, len(body) - length + 1))
        texts.append(body[start : start + length].ljust(length))
        labels.append(speaker)
    return texts, labels
