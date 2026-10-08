import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import audio


def write_wav(path: Path, samples: list[float]) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(audio.RATE)
        file.writeframes(array("h", [int(32767 * s) for s in samples]).tobytes())


class AudioTest(unittest.TestCase):
    def test_read_wav(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            write_wav(Path(folder) / "a.wav", [0.0, 0.5, -0.5])
            self.assertEqual([round(v, 3) for v in audio.read_wav(Path(folder) / "a.wav")], [0.0, 0.5, -0.5])

    def test_mel_filters_cover_the_spectrum(self) -> None:
        filters = audio.mel_filters()
        self.assertEqual((len(filters), len(filters[0])), (audio.BANDS, audio.FFT // 2 + 1))
        for band in filters:
            self.assertAlmostEqual(max(band), 1.0, delta=0.35)    # each peaks near 1 at its centre bin
        centres = [max(range(len(band)), key=band.__getitem__) for band in filters]
        self.assertEqual(centres, sorted(centres))                   # bands rise in frequency

    def test_a_pure_tone_lights_up_its_band(self) -> None:
        spectrogram = audio.Spectrogram()
        filters = audio.mel_filters()
        for hz in (300.0, 1500.0, 3000.0):
            tone = [0.5 * math.sin(2 * math.pi * hz * n / audio.RATE) for n in range(audio.RATE)]
            values = spectrogram(tone)
            loudest = max(range(audio.BANDS), key=lambda b: values[b * audio.FRAMES + 50])
            bin_index = round(hz * audio.FFT / audio.RATE)
            self.assertEqual(loudest, max(range(audio.BANDS), key=lambda b: filters[b][bin_index]))

    def test_dft_matches_a_direct_sum(self) -> None:
        spectrogram = audio.Spectrogram()
        frame = [math.sin(0.3 * n) + 0.2 * math.cos(1.1 * n) for n in range(audio.WINDOW)]
        hann = [0.5 - 0.5 * math.cos(2 * math.pi * n / audio.WINDOW) for n in range(audio.WINDOW)]
        half = audio.FFT // 2 + 1
        for k in (0, 7, 64, 128):
            real = sum(hann[n] * frame[n] * math.cos(2 * math.pi * k * n / audio.FFT) for n in range(audio.WINDOW))
            self.assertAlmostEqual(sum(frame[n] * spectrogram.cos.data[n * half + k] for n in range(audio.WINDOW)),
                                   real, places=10)

    def test_training_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as folder:
            for digit in range(10):                       # a tone per digit, 7 clips each (5 test, 2 train)
                for index in range(7):
                    hz = 200 + 300 * digit + 10 * index
                    write_wav(Path(folder) / f"{digit}_tone_{index}.wav",
                              [0.3 * math.sin(2 * math.pi * hz * n / audio.RATE) for n in range(3000)])
            with contextlib.redirect_stdout(io.StringIO()):
                result = audio.main(["--data", folder, "--frameworks", "nanograd", "torch", "--epochs", "2",
                                     "--batch-size", "5"])
        self.assertEqual(result["nanograd"]["accuracy"], result["torch"]["accuracy"])
        self.assertLess(result["max_weight_difference"], 1e-10)


if __name__ == "__main__":
    unittest.main()
