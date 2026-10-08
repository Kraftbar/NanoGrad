import contextlib
import io
import math
import struct
import tempfile
import unittest
from pathlib import Path

import sr


def write_bmp(path: Path, rows: list[list[tuple[int, int, int]]]) -> None:
    """A 24-bit bottom-up BMP."""

    height, width = len(rows), len(rows[0])
    stride = (3 * width + 3) // 4 * 4
    pixels = b"".join(bytes(v for r, g, b in row for v in (b, g, r)).ljust(stride, b"\0") for row in rows[::-1])
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    path.write_bytes(header + info + pixels)


def pattern(height: int, width: int) -> list[list[tuple[int, int, int]]]:
    return [[((x * 37 + y * 11) % 256, (x * 5 + y * 23) % 256, (x * y) % 256) for x in range(width)]
            for y in range(height)]


class SuperResolutionTest(unittest.TestCase):
    def test_resize_weights_sum_to_one_and_keep_constants(self) -> None:
        for size, scale in ((12, 1 / 3), (4, 3), (9, 2)):
            for taps in sr.resize_weights(size, scale):
                self.assertAlmostEqual(sum(w for _, w in taps), 1.0)
                self.assertTrue(all(0 <= i < size for i, _ in taps))
        flat = [[0.25] * 9 for _ in range(6)]
        for row in sr.imresize(flat, 1 / 3) + sr.imresize(flat, 3):
            for value in row:
                self.assertAlmostEqual(value, 0.25)

    def test_upscaling_interpolates_a_linear_ramp(self) -> None:
        # A cubic kernel with a = -0.5 reproduces linear functions away from the borders.
        ramp = [[x / 10 for x in range(10)]]
        out = sr.imresize(ramp * 4, 2)[0]
        for x in range(4, 16):
            u = (x + 1) / 2 + 0.5 * (1 - 1 / 2) - 1          # output centre in 0-based input coordinates
            self.assertAlmostEqual(out[x], u / 10)

    def test_psnr(self) -> None:
        a = [[0.0, 0.0], [0.0, 0.0]]
        b = [[1 / 255, 0.0], [0.0, 0.0]]
        self.assertAlmostEqual(sr.psnr(a, b, 0), 10 * math.log10(255 ** 2 / 0.25))

    def test_training_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as folder:
            for name, size in (("train", (40, 46)), ("set5", (30, 27))):
                (Path(folder) / name).mkdir()
                write_bmp(Path(folder) / name / "x.bmp", pattern(*size))
            with contextlib.redirect_stdout(io.StringIO()):
                result = sr.main(["--data", folder, "--frameworks", "nanograd", "torch", "--epochs", "2",
                                  "--test-sets", "set5", "--batch-size", "2"])
        self.assertAlmostEqual(result["nanograd"]["psnr"]["set5"], result["torch"]["psnr"]["set5"], places=6)
        self.assertLess(result["max_weight_difference"], 1e-10)


if __name__ == "__main__":
    unittest.main()
