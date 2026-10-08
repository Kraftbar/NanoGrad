import random
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import images


def make_png(rows: list[list], depth: int, filters: list[int]) -> bytes:
    """A greyscale PNG of the given bit depth, or 8-bit RGB if the pixels are
    tuples; row y uses filter filters[y] (0: none, 2: up, 1: left)."""

    width, top = len(rows[0]), (1 << depth) - 1
    colour = isinstance(rows[0][0], tuple)
    stride = 3 * width if colour else (width * depth + 7) // 8
    raw, previous = b"", bytes(stride)
    for row, kind in zip(rows, filters):
        line = bytearray(stride)
        if colour:
            line = bytearray(v for pixel in row for v in pixel)
        for x, value in enumerate([] if colour else row):
            line[(x * depth) // 8] |= (value * top // 255) << (8 - depth - (x * depth) % 8)
        if kind == 1:
            step = 3 if colour else 1
            encoded = bytes((line[i] - (line[i - step] if i >= step else 0)) & 255 for i in range(stride))
            raw += bytes([1]) + encoded
            previous = bytes(line)
            continue
        encoded = bytes((line[i] - previous[i]) & 255 for i in range(stride)) if kind == 2 else bytes(line)
        raw += bytes([kind]) + encoded
        previous = bytes(line)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, len(rows), depth, 2 if colour else 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def write_bmp(path: Path, rows: list[list[tuple[int, int, int]]]) -> None:
    """A 24-bit bottom-up BMP."""

    height, width = len(rows), len(rows[0])
    stride = (3 * width + 3) // 4 * 4
    pixels = b"".join(bytes(v for r, g, b in row for v in (b, g, r)).ljust(stride, b"\0") for row in rows[::-1])
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    path.write_bytes(header + info + pixels)


class ImageFileTest(unittest.TestCase):
    def test_png_grey_and_binary(self) -> None:
        rng = random.Random(0)
        binary = [[255 * rng.randrange(2) for _ in range(11)] for _ in range(5)]
        self.assertEqual(images.read_png(make_png(binary, 1, [0, 2, 2, 0, 2])), binary)
        grey = [[rng.randrange(256) for _ in range(6)] for _ in range(4)]
        self.assertEqual(images.read_png(make_png(grey, 8, [2, 0, 1, 2])), grey)

    def test_png_rgb(self) -> None:
        rng = random.Random(1)
        rgb = [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(5)] for _ in range(4)]
        self.assertEqual(images.read_png(make_png(rgb, 8, [0, 1, 2, 1])), rgb)

    def test_bmp_round_trip(self) -> None:
        rows = [[((x * 37 + y * 11) % 256, (x * 5) % 256, (x * y) % 256) for x in range(7)] for y in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.bmp"
            write_bmp(path, rows)                         # 21 bytes per row: exercises the padding
            self.assertEqual(images.read_bmp(path), rows)


if __name__ == "__main__":
    unittest.main()
