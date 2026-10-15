"""Live keyword spotting from the microphone, with a model trained by kws.py.

kws.py classifies one-second clips. Listening is a stream, so the
detector slides a one-second window over it, one hop (125 ms) at a
time, and classifies each window with the same MFCCs, statistics and
DS-CNN as training. A word is about half a second long, so it sits
inside several windows in a row; turning those into one printed word
takes three rules:

    gate        a window whose loudest hop is not clearly above the noise floor
                (the quietest hop of the last few seconds) is silence, without
                running the model, so a quiet room costs almost nothing
    smoothing   the class probabilities are averaged over the last 3 windows,
                so one odd window does not fire
    threshold   a keyword prints when its smoothed probability is at least 0.7,
    refractory  and then nothing prints for 1 s, nor until that word's probability
                has fallen below the threshold, so one utterance prints once

The microphone is read with ALSA's ``arecord`` (16 kHz, 16-bit mono raw
samples on its standard output); ``--wav`` runs the same detector over a
16 kHz WAV file, as fast as it can, and ``--stdin`` reads raw samples.

    python3 kws.py --epochs 36 --dropout --l2 1e-4 --save runs/kws/model.json
    python3 kws_live.py                                   # the default microphone
    python3 kws_live.py --device plughw:1,0 --verbose      # a USB webcam's; show every window
    python3 kws_live.py --make-test-wav runs/kws/test.wav  # test clips with silence in between
    python3 kws_live.py --wav runs/kws/test.wav
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import wave
from array import array
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator, NamedTuple

from audio import read_wav
from kws import CLASSES, KEYWORDS, MFCC, RATE, load_checkpoint, normalize, one_second, probabilities

HOP = RATE // 8                       # 125 ms between windows


class Detection(NamedTuple):
    seconds: float                    # the end of the window that fired, from the start of the stream
    word: str
    confidence: float                 # the smoothed probability


class KeywordModel:
    """One second of samples -> class probabilities, through kws.py's front end and a saved DS-CNN."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.model, self.statistics = load_checkpoint(checkpoint)
        self.mfcc = MFCC()

    def __call__(self, window: list[float]) -> list[float]:
        (features,) = self.mfcc([window])
        return probabilities(self.model, normalize(features, self.statistics), 1)


def level(samples: list[float]) -> float:
    """Root mean square, about 0.01 for a quiet room through a webcam microphone and 0.05-0.3 for speech."""

    return math.sqrt(sum(v * v for v in samples) / len(samples))


class Detector:
    """Feed it samples as they arrive; it returns the keywords heard (usually none).

    ``classify`` maps a one-second window to probabilities over ``CLASSES``. The
    rules (gate, smoothing, threshold, refractory period) are described at the top
    of this file."""

    def __init__(self, classify: Callable[[list[float]], list[float]], *, hop: int = HOP, smooth: int = 3,
                 threshold: float = 0.7, refractory: float = 1.0, gate_db: float = 6.0, min_level: float = 0.002,
                 floor_seconds: float = 4.0, on_window: Callable | None = None) -> None:
        self.classify, self.hop, self.threshold, self.refractory = classify, hop, threshold, refractory
        self.gate_ratio, self.min_level = 10 ** (gate_db / 20), min_level      # 6 dB: twice the floor's level
        self.on_window = on_window                       # called with (seconds, level, probabilities or None)
        self.window = [0.0] * RATE                       # the last second; starts as digital silence
        self.pending: list[float] = []                   # samples not yet a whole hop
        self.hop_levels: deque[float] = deque(maxlen=RATE // hop)         # the window's hops
        self.floor_levels: deque[float] = deque(maxlen=max(1, round(floor_seconds * RATE / hop)))
        self.recent: deque[list[float]] = deque(maxlen=smooth)
        self.silence = [1.0 if name == "_silence_" else 0.0 for name in CLASSES]
        self.samples_seen, self.quiet_until, self.model_calls = 0, 0.0, 0
        self.last_word: int | None = None                # the class that fired, until it fades

    def feed(self, samples: list[float]) -> list[Detection]:
        detections = []
        self.pending.extend(samples)
        while len(self.pending) >= self.hop:
            step, self.pending = self.pending[: self.hop], self.pending[self.hop :]
            self.window = self.window[self.hop :] + step
            self.samples_seen += self.hop
            detection = self._step(level(step))
            if detection:
                detections.append(detection)
        return detections

    def _step(self, hop_level: float) -> Detection | None:
        self.hop_levels.append(hop_level)
        self.floor_levels.append(hop_level)
        seconds = self.samples_seen / RATE
        gate = max(self.min_level, self.gate_ratio * min(self.floor_levels))
        if max(self.hop_levels) < gate:
            estimate = None                              # silence: skip the model
        else:
            estimate = self.classify(self.window)
            self.model_calls += 1
        if self.on_window:
            self.on_window(seconds, max(self.hop_levels), estimate)
        self.recent.append(estimate or self.silence)     # a gated window counts as certain silence
        smoothed = [sum(p[c] for p in self.recent) / len(self.recent) for c in range(len(CLASSES))]
        if self.last_word is not None and smoothed[self.last_word] < self.threshold:
            self.last_word = None                        # re-armed
        best = max(range(len(CLASSES)), key=lambda c: smoothed[c])
        if (CLASSES[best] in KEYWORDS and smoothed[best] >= self.threshold and seconds >= self.quiet_until
                and best != self.last_word):
            self.quiet_until, self.last_word = seconds + self.refractory, best
            return Detection(seconds, CLASSES[best], smoothed[best])
        return None


def raw_chunks(stream, samples: int = HOP) -> Iterator[list[float]]:
    """16-bit little-endian mono samples from a byte stream, scaled to [-1, 1), ``samples`` at a time."""

    while True:
        data = stream.read(2 * samples)
        if len(data) < 2:
            return
        chunk = array("h", data[: len(data) // 2 * 2])
        if sys.byteorder == "big":
            chunk.byteswap()
        yield [s / 32768.0 for s in chunk]


def wav_chunks(path: str | Path, samples: int = HOP) -> Iterator[list[float]]:
    """A 16 kHz WAV file in hop-sized pieces, as if it were streaming, then a second of silence
    so a word at the very end still fills a window."""

    with wave.open(str(path)) as file:
        if file.getframerate() != RATE:
            raise ValueError(f"{path}: expected {RATE} Hz, got {file.getframerate()}")
    clip = read_wav(path) + [0.0] * RATE
    for first in range(0, len(clip), samples):
        yield clip[first : first + samples]


def make_test_wav(folder: str | Path, path: str | Path, words: list[str], gap: float = 1.0, noise: float = 0.0,
                  seed: int = 0) -> list[float]:
    """One random test-set clip of each word (a keyword, or "_unknown_" for some other word), with
    ``gap`` seconds of silence between them; ``noise`` mixes in background noise at that volume.
    Returns the second at which each clip starts."""

    folder, rng = Path(folder), random.Random(seed)
    samples, starts = [0.0] * int(gap * RATE), []
    for word in words:
        starts.append(len(samples) / RATE)
        clips = sorted((folder / "test_set" / word).glob("*.wav"))
        samples += one_second(read_wav(rng.choice(clips))) + [0.0] * int(gap * RATE)
    if noise:
        recording = read_wav(sorted((folder / "_background_noise_").glob("*.wav"))[rng.randrange(6)])
        samples = [v + noise * recording[n % len(recording)] for n, v in enumerate(samples)]
    pcm = array("h", [max(-32768, min(32767, round(v * 32768))) for v in samples])
    if sys.byteorder == "big":
        pcm.byteswap()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(pcm.tobytes())
    return starts


def main(argv: list[str] | None = None) -> list[Detection]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="runs/kws/model.json", help="a checkpoint from kws.py --save")
    parser.add_argument("--device", default="default", help="ALSA capture device, e.g. plughw:1,0 (arecord -l)")
    parser.add_argument("--wav", type=Path, help="run over a 16 kHz mono WAV file instead of the microphone")
    parser.add_argument("--stdin", action="store_true", help="read raw S16_LE 16 kHz mono samples from stdin")
    parser.add_argument("--seconds", type=float, help="stop listening after this long")
    parser.add_argument("--threshold", type=float, default=0.7, help="smoothed probability needed to print a word")
    parser.add_argument("--smooth", type=int, default=3, help="windows averaged")
    parser.add_argument("--refractory", type=float, default=1.0, help="seconds of quiet after a detection")
    parser.add_argument("--gate-db", type=float, default=6.0, help="how far above the noise floor speech must be")
    parser.add_argument("--verbose", action="store_true", help="print every window's level and best class")
    parser.add_argument("--make-test-wav", type=Path, help="write a test WAV of test-set clips and exit")
    parser.add_argument("--words", nargs="+", default=["yes", "no", "_unknown_", "up", "down", "left", "right",
                                                         "_unknown_", "on", "off", "stop", "go"])
    parser.add_argument("--noise", type=float, default=0.0, help="background noise volume for --make-test-wav")
    parser.add_argument("--data", default="data/speech_commands")
    args = parser.parse_args(argv)

    if args.make_test_wav:
        starts = make_test_wav(args.data, args.make_test_wav, args.words, noise=args.noise)
        print(f"wrote {args.make_test_wav}: " + ", ".join(f"{w} at {s:.0f} s" for w, s in zip(args.words, starts)))
        return []

    def show(seconds: float, loudest: float, estimate: list[float] | None) -> None:
        if estimate is None:
            print(f"  {seconds:7.2f} s  level {loudest:.4f}  (gated)")
        else:
            best = max(range(len(CLASSES)), key=lambda c: estimate[c])
            print(f"  {seconds:7.2f} s  level {loudest:.4f}  {CLASSES[best]} {estimate[best]:.2f}")

    detector = Detector(KeywordModel(args.model), smooth=args.smooth, threshold=args.threshold,
                        refractory=args.refractory, gate_db=args.gate_db, on_window=show if args.verbose else None)
    recorder = None
    if args.wav:
        chunks = wav_chunks(args.wav)
    elif args.stdin:
        chunks = raw_chunks(sys.stdin.buffer)
    else:
        recorder = subprocess.Popen(["arecord", "-q", "-D", args.device, "-f", "S16_LE", "-r", str(RATE), "-c", "1",
                                     "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunks = raw_chunks(recorder.stdout)
        print(f"listening on {args.device} for: {' '.join(KEYWORDS)} (Ctrl-C to stop)", flush=True)
    heard, start, stopped = [], perf_counter(), False
    try:
        for chunk in chunks:
            for detection in detector.feed(chunk):
                heard.append(detection)
                print(f"{detection.seconds:7.2f} s  {detection.word:5s}  ({detection.confidence:.2f})", flush=True)
            if args.seconds and detector.samples_seen >= args.seconds * RATE:
                stopped = True
                break
    except KeyboardInterrupt:
        stopped = True
    finally:
        if recorder:
            recorder.terminate()
            errors = recorder.communicate()[1].decode().strip()
            if errors and not stopped:                   # arecord quit by itself: a busy or missing device
                print(errors, file=sys.stderr)
    seconds = detector.samples_seen / RATE
    print(f"{seconds:.1f} s of audio in {perf_counter() - start:.1f} s, {len(heard)} keywords, "
          f"the model ran on {detector.model_calls} of {round(seconds * RATE / detector.hop)} windows", flush=True)
    return heard


if __name__ == "__main__":
    main()
