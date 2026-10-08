import random
import struct
import unittest
import zlib

import fewshot


def make_png(rows: list[list[int]], depth: int, filters: list[int]) -> bytes:
    """A greyscale PNG of the given bit depth; row y uses filter filters[y] (0: none, 2: up)."""

    width, top = len(rows[0]), (1 << depth) - 1
    stride = (width * depth + 7) // 8
    raw, previous = b"", bytes(stride)
    for row, kind in zip(rows, filters):
        line = bytearray(stride)
        for x, value in enumerate(row):
            line[(x * depth) // 8] |= (value * top // 255) << (8 - depth - (x * depth) % 8)
        encoded = bytes((line[i] - previous[i]) & 255 for i in range(stride)) if kind == 2 else bytes(line)
        raw += bytes([kind]) + encoded
        previous = bytes(line)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, len(rows), depth, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class FewShotTest(unittest.TestCase):
    def test_read_png(self) -> None:
        rng = random.Random(0)
        binary = [[255 * rng.randrange(2) for _ in range(11)] for _ in range(5)]
        self.assertEqual(fewshot.read_png(make_png(binary, 1, [0, 2, 2, 0, 2])), binary)
        grey = [[rng.randrange(256) for _ in range(6)] for _ in range(4)]
        self.assertEqual(fewshot.read_png(make_png(grey, 8, [2, 0, 2, 2])), grey)

    def test_rotate_and_shrink(self) -> None:
        image = [[1, 2], [3, 4]]
        self.assertEqual(fewshot.rotate(image, 90), [[2, 4], [1, 3]])       # counter-clockwise
        self.assertEqual(fewshot.rotate(image, 360), image)
        page = [[255] * 56 for _ in range(56)]
        page[1][1] = 0                    # ink where output pixel (0, 0) samples: (0 + 0.5) * 56 / 28 = 1
        small = fewshot.shrink(page)
        self.assertEqual((len(small), small[0], sum(small)), (28 * 28, 1, 1))

    def test_scores_are_negative_squared_distances_to_prototypes(self) -> None:
        runner = fewshot.NanoGradRunner(fewshot.encoder(channels=4), 1e-3)
        rng = random.Random(1)
        classes = {f"c{n}": [[rng.randrange(2) for _ in range(28 * 28)] for _ in range(4)] for n in range(3)}
        support, queries, way = fewshot.episode_batch(classes, 3, 2, 1, rng)
        runner.model.eval()
        scores = runner._scores(support, queries, way).data
        vectors = runner.embed(support + queries)
        prototypes = [[(a + b) / 2 for a, b in zip(vectors[2 * n], vectors[2 * n + 1])] for n in range(3)]
        for q in range(3):
            for n in range(3):
                distance = sum((a - b) ** 2 for a, b in zip(vectors[6 + q], prototypes[n]))
                self.assertAlmostEqual(scores[q * 3 + n], -distance, places=9)

    def test_training_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        rng = random.Random(2)
        classes = {f"c{n}": [[rng.randrange(2) for _ in range(28 * 28)] for _ in range(4)] for n in range(5)}
        model = fewshot.encoder(channels=8)
        runners = [fewshot.NanoGradRunner(model, 1e-3), fewshot.TorchRunner(model, 1e-3)]
        for step in range(3):
            batch = fewshot.episode_batch(classes, 4, 1, 2, random.Random(step))
            losses = [runner.train_step(*batch, 1e-3)[0] for runner in runners]
            self.assertAlmostEqual(losses[0], losses[1], places=9)
        difference = max(abs(a - b) for x, y in zip(runners[0].weights(), runners[1].weights()) for a, b in zip(x, y))
        self.assertLess(difference, 1e-10)


if __name__ == "__main__":
    unittest.main()
