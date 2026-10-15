import io
import tempfile
import unittest
from array import array
from pathlib import Path

import kws
import kws_live

RATE = kws.RATE


def one_hot(name: str) -> list[float]:
    return [1.0 if c == name else 0.0 for c in kws.CLASSES]


def burst(seconds: float, volume: float = 0.3) -> list[float]:
    """A stand-in for a spoken word: a loud alternating signal."""

    return [volume if n % 2 else -volume for n in range(int(seconds * RATE))]


def quiet(seconds: float) -> list[float]:
    return [0.0] * int(seconds * RATE)


class FakeModel:
    """Says "yes" when at least 0.4 s of the window is loud, "_unknown_" otherwise, and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, window: list[float]) -> list[float]:
        self.calls += 1
        loud = sum(abs(v) > 0.1 for v in window)
        return one_hot("yes" if loud >= 0.4 * RATE else "_unknown_")


class DetectorTest(unittest.TestCase):
    def words(self, samples, model=None, **options):
        detector = kws_live.Detector(model or FakeModel(), **options)
        return [(d.word, d.seconds) for d in detector.feed(samples)]

    def test_silence_never_runs_the_model(self) -> None:
        model = FakeModel()
        detector = kws_live.Detector(model)
        self.assertEqual(detector.feed(quiet(3.0)), [])
        self.assertEqual(model.calls, 0)

    def test_one_word_prints_once(self) -> None:
        heard = self.words(quiet(1.0) + burst(0.6) + quiet(2.0))
        self.assertEqual([word for word, _ in heard], ["yes"])
        # Windows end every 0.125 s; from 1.5 s on they hold 0.4 s of the word, and the third such
        # window (1.75 s) brings the smoothed probability from 2/3 to 1.
        self.assertAlmostEqual(heard[0][1], 1.75)

    def test_refractory_period_prints_close_words_once_and_far_ones_twice(self) -> None:
        close = quiet(1.0) + burst(0.6) + quiet(0.3) + burst(0.6) + quiet(2.0)
        self.assertEqual(len(self.words(close)), 1)
        # A word stays in the window for over a second: it prints once even after the refractory
        # period, until the detector re-arms (the word's probability falls below the threshold).
        self.assertEqual(len(self.words(quiet(1.0) + burst(2.5) + quiet(2.0), refractory=0.3)), 1)
        far = quiet(1.0) + burst(0.6) + quiet(1.5) + burst(0.6) + quiet(2.0)
        self.assertEqual([word for word, _ in self.words(far)], ["yes", "yes"])

    def test_smoothing_ignores_a_single_window(self) -> None:
        class OneWindow:                        # "yes" for exactly one window
            def __init__(self):
                self.calls = 0

            def __call__(self, window):
                self.calls += 1
                return one_hot("yes" if self.calls == 3 else "_unknown_")

        samples = quiet(1.0) + burst(2.0)
        self.assertEqual(self.words(samples, OneWindow(), smooth=3, threshold=0.7), [])
        self.assertEqual(len(self.words(samples, OneWindow(), smooth=1, threshold=0.7)), 1)

    def test_unknown_and_silence_never_print(self) -> None:
        class Always:
            def __init__(self, name):
                self.name = name

            def __call__(self, window):
                return one_hot(self.name)

        for name in ("_unknown_", "_silence_"):
            self.assertEqual(self.words(burst(3.0), Always(name)), [])

    def test_the_gate_follows_the_noise_floor(self) -> None:
        model = FakeModel()
        noisy = [0.05 * (1 if n % 3 else -1) for n in range(3 * RATE)]       # steady noise, far above min_level
        detector = kws_live.Detector(model, gate_db=6.0)
        detector.feed(noisy)                    # steady noise is the floor, so it is gated
        self.assertEqual(model.calls, 0)
        self.assertEqual([d.word for d in detector.feed(burst(0.6) + noisy)], ["yes"])

    def test_chunk_sizes_do_not_matter(self) -> None:
        samples = quiet(0.5) + burst(0.6) + quiet(1.5) + burst(0.5) + quiet(1.5)
        whole = self.words(samples)
        detector, pieces = kws_live.Detector(FakeModel()), []
        for first in range(0, len(samples), 777):
            pieces += [(d.word, d.seconds) for d in detector.feed(samples[first : first + 777])]
        self.assertEqual(pieces, whole)
        self.assertEqual(len(whole), 2)

    def test_raw_chunks_decode_16_bit_samples(self) -> None:
        pcm = array("h", [0, 16384, -32768, 32767, 1])
        stream = io.BytesIO(pcm.tobytes())
        chunks = list(kws_live.raw_chunks(stream, samples=2))
        self.assertEqual(chunks, [[0.0, 0.5], [-1.0, 32767 / 32768], [1 / 32768]])


class CheckpointTest(unittest.TestCase):
    def test_saved_model_gives_the_same_probabilities(self) -> None:
        model = kws.ds_cnn(seed=3)
        statistics = [(0.1 * c, 1.0 + c) for c in range(kws.COEFFICIENTS)]
        window = burst(1.0, 0.05)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            kws.save_checkpoint(path, model, statistics)
            live = kws_live.KeywordModel(path)
        (features,) = kws.MFCC()([window])
        expected = kws.probabilities(model, kws.normalize(features, statistics), 1)
        self.assertEqual(live.statistics, statistics)
        for a, b in zip(live(window), expected):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(sum(expected), 1.0)


if __name__ == "__main__":
    unittest.main()
