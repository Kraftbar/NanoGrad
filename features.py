"""Corner detection and patch matching on grayscale images.

Images are 2D ``Tensor`` objects of shape (height, width) with values in
[0, 1], so the filtering below is NanoGrad's own ``conv2d_valid`` and
``avg_pool2d`` (run under ``no_grad``: nothing here needs gradients).

Harris corners
--------------
A corner is a spot where the image changes in *every* direction. With image
gradients Ix and Iy, the structure tensor summed over a small window is

    M = [ sum Ix*Ix   sum Ix*Iy ]
        [ sum Ix*Iy   sum Iy*Iy ]

Flat regions have two small eigenvalues, edges one large one, and corners
two large ones. Harris scores this without an eigen solve:

    R = det(M) - k * trace(M)^2        (k ~ 0.04 .. 0.06)

The peak of R sits slightly inside a sharp corner (about 1.4 px for an L
corner with a 5x5 window). That bias is the same in every view of the
corner, so matched positions still agree closely.

Matching
--------
Each corner is described by the pixels of the patch around it, made
zero-mean and unit-length so that brightness and contrast changes cancel.
The dot product of two such vectors is their normalized cross-correlation
(NCC, 1 = identical). Matches must be mutual best matches and pass Lowe's
ratio test: clearly better than the second-best candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tensor import Tensor, avg_pool2d, conv2d_valid, no_grad


SOBEL_X = Tensor.from_list([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
])
SOBEL_Y = Tensor.from_list([
    [-1.0, -2.0, -1.0],
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 1.0],
])


@dataclass(frozen=True)
class Corner:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class Match:
    first: int     # index into the first corner list
    second: int    # index into the second corner list
    score: float   # NCC of the two patches


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------


def harris_response(image: Tensor, *, window: int = 5, k: float = 0.05) -> tuple[Tensor, int]:
    """Harris score map and its pixel offset into the original image.

    Each valid convolution trims the border, so score (row, col) belongs to
    image pixel (row + offset, col + offset).
    """

    if window % 2 == 0:
        raise ValueError("harris window must be odd so corners land on pixels")
    with no_grad():
        ix = conv2d_valid(image, SOBEL_X)
        iy = conv2d_valid(image, SOBEL_Y)
        # Window sums, written as means (the constant factor does not move
        # corners; it only rescales R).
        sxx = avg_pool2d(ix * ix, (window, window), stride=(1, 1))
        syy = avg_pool2d(iy * iy, (window, window), stride=(1, 1))
        sxy = avg_pool2d(ix * iy, (window, window), stride=(1, 1))
        trace = sxx + syy
        response = sxx * syy - sxy * sxy - trace * trace * k
    offset = 1 + window // 2
    return response, offset


def harris_corners(
    image: Tensor,
    *,
    max_corners: int = 200,
    min_distance: int = 6,
    relative_threshold: float = 0.01,
    window: int = 5,
    k: float = 0.05,
) -> list[Corner]:
    """Strongest Harris corners, at least ``min_distance`` pixels apart.

    Non-maximum suppression is greedy: take corners from strongest to
    weakest and skip any that land too close to one already taken (checked
    on whole pixels, before sub-pixel refinement).
    """

    response, offset = harris_response(image, window=window, k=k)
    rows, cols = response.shape
    scores = response.data
    floor = relative_threshold * max(scores)
    candidates = sorted(
        (
            (score, index)
            for index, score in enumerate(scores)
            if score > floor and _is_local_peak(scores, index, rows, cols)
        ),
        reverse=True,
    )

    corners: list[Corner] = []
    taken: list[tuple[int, int]] = []  # whole-pixel positions of accepted corners
    for score, index in candidates:
        row, col = divmod(index, cols)
        x, y = col + offset, row + offset
        if all((x - tx) ** 2 + (y - ty) ** 2 >= min_distance ** 2 for tx, ty in taken):
            dx, dy = _subpixel_offset(scores, row, col, rows, cols)
            taken.append((x, y))
            corners.append(Corner(x + dx, y + dy, score))
            if len(corners) == max_corners:
                break
    return corners


def _is_local_peak(scores: list[float], index: int, rows: int, cols: int) -> bool:
    row, col = divmod(index, cols)
    value = scores[index]
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            r, c = row + d_row, col + d_col
            if (d_row or d_col) and 0 <= r < rows and 0 <= c < cols and scores[r * cols + c] > value:
                return False
    return True


def _subpixel_offset(scores: list[float], row: int, col: int, rows: int, cols: int) -> tuple[float, float]:
    """Refine a peak by fitting a parabola through it and its two neighbours.

    For samples (left, center, right) the parabola's vertex sits at
    (left - right) / (2 * (left - 2 center + right)) from the center.
    """

    def vertex(left: float, center: float, right: float) -> float:
        curvature = left - 2.0 * center + right
        if curvature >= 0.0:
            return 0.0
        return max(-0.5, min(0.5, (left - right) / (2.0 * curvature)))

    center = scores[row * cols + col]
    dx = 0.0
    dy = 0.0
    if 0 < col < cols - 1:
        dx = vertex(scores[row * cols + col - 1], center, scores[row * cols + col + 1])
    if 0 < row < rows - 1:
        dy = vertex(scores[(row - 1) * cols + col], center, scores[(row + 1) * cols + col])
    return dx, dy


# ---------------------------------------------------------------------------
# Descriptors and matching
# ---------------------------------------------------------------------------


def patch_descriptor(image: Tensor, corner: Corner, *, radius: int = 5) -> list[float] | None:
    """Zero-mean, unit-length patch around a corner, or None near the border."""

    height, width = image.shape
    x, y = round(corner.x), round(corner.y)
    if x < radius or y < radius or x >= width - radius or y >= height - radius:
        return None

    values = [
        image.data[row * width + col]
        for row in range(y - radius, y + radius + 1)
        for col in range(x - radius, x + radius + 1)
    ]
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    length = math.sqrt(sum(value * value for value in centered))
    if length < 1e-6:
        return None
    return [value / length for value in centered]


def match_descriptors(
    first: list[list[float] | None],
    second: list[list[float] | None],
    *,
    ratio: float = 0.9,
    min_score: float = 0.7,
) -> list[Match]:
    """Mutual nearest neighbours by NCC, filtered by Lowe's ratio test.

    For unit vectors, squared distance is 2 - 2 * NCC, so "nearest" means
    "highest NCC". The ratio test compares those distances: the best match
    must be closer than ``ratio`` times the runner-up.
    """

    def best_two(descriptor: list[float], others: list[list[float] | None]) -> tuple[int, float, float]:
        best_index, best, second_best = -1, -2.0, -2.0
        for index, other in enumerate(others):
            if other is None:
                continue
            score = sum(a * b for a, b in zip(descriptor, other))
            if score > best:
                best_index, best, second_best = index, score, best
            elif score > second_best:
                second_best = score
        return best_index, best, second_best

    backward = {}
    for index, descriptor in enumerate(second):
        if descriptor is not None:
            backward[index] = best_two(descriptor, first)[0]

    matches = []
    for index, descriptor in enumerate(first):
        if descriptor is None:
            continue
        partner, best, second_best = best_two(descriptor, second)
        if partner < 0 or best < min_score or backward.get(partner) != index:
            continue
        if (2.0 - 2.0 * best) >= ratio * ratio * (2.0 - 2.0 * second_best):
            continue
        matches.append(Match(index, partner, best))
    return matches


# ---------------------------------------------------------------------------
# PGM files (the simplest grayscale image format)
# ---------------------------------------------------------------------------


def save_pgm(path: str | Path, image: Tensor) -> None:
    """Write an 8-bit binary PGM ("P5") file; most image viewers open it."""

    height, width = image.shape
    pixels = bytes(max(0, min(255, round(value * 255))) for value in image.data)
    Path(path).write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)


def save_png(path: str | Path, image: Tensor) -> None:
    """Write a grayscale PNG (values in [0, 1]) with only the standard library.

    A PNG is a signature, then chunks of (length, type, data, CRC32): IHDR
    (size, 8-bit grayscale), IDAT (zlib-compressed rows, each prefixed by a
    0 byte meaning "no filter"), and IEND.
    """

    import struct
    import zlib

    height, width = image.shape
    rows = b"".join(
        b"\x00" + bytes(max(0, min(255, round(value * 255))) for value in image.data[r * width : (r + 1) * width])
        for r in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


def load_pgm(path: str | Path) -> Tensor:
    """Read an 8-bit binary PGM file into a (height, width) Tensor in [0, 1].

    The header is four whitespace-separated fields (magic, width, height,
    max value), with ``#`` comments running to the end of a line allowed
    anywhere between them, followed by one whitespace byte and the pixels.
    """

    raw = Path(path).read_bytes()
    fields: list[bytes] = []
    position = 0
    while len(fields) < 4:
        if position >= len(raw):
            raise ValueError("PGM header is truncated")
        byte = raw[position:position + 1]
        if byte.isspace():
            position += 1
        elif byte == b"#":
            end = raw.find(b"\n", position)
            position = len(raw) if end < 0 else end + 1
        else:
            start = position
            while position < len(raw) and not raw[position:position + 1].isspace() \
                    and raw[position:position + 1] != b"#":
                position += 1
            fields.append(raw[start:position])
    magic, width, height, max_value = fields[0], int(fields[1]), int(fields[2]), int(fields[3])
    if magic != b"P5" or not 0 < max_value <= 255:
        raise ValueError("load_pgm reads 8-bit binary (P5) PGM files only")
    pixels = raw[position + 1 : position + 1 + width * height]
    if len(pixels) != width * height:
        raise ValueError("PGM pixel data is truncated")
    return Tensor([value / max_value for value in pixels], (height, width))
