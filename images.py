"""Image files without dependencies: PNG and BMP decoders.

Only what the datasets here need: non-interlaced PNGs (greyscale at 1 to 8
bits, or 8-bit RGB) and uncompressed BMPs (24-bit, or 8-bit with a
palette). Both return rows of pixels. ``read_png`` gives an int (0-255)
per pixel for a grey image and an (R, G, B) tuple for a colour one;
``read_bmp`` always gives tuples (a grey palette repeats the value).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def read_png(raw: bytes) -> list[list]:
    """Rows of pixels from a PNG's bytes.

    A PNG is chunks; the IDAT chunks together are one zlib stream of rows,
    each starting with a filter byte that says how the row was predicted
    from the pixel to its left, the one above, or both (the decoder adds
    the prediction back)."""

    width, height, depth, color = struct.unpack(">IIBB", raw[16:26])
    channels = {0: 1, 2: 3}.get(color)
    if channels is None or depth > 8 or (channels == 3 and depth != 8):
        raise ValueError("only greyscale and 8-bit RGB PNGs are supported")
    position, compressed = 8, b""
    while position < len(raw):                       # chunks: length, type, data, CRC
        length = struct.unpack(">I", raw[position : position + 4])[0]
        if raw[position + 4 : position + 8] == b"IDAT":
            compressed += raw[position + 8 : position + 8 + length]
        position += 12 + length
    data = zlib.decompress(compressed)
    stride = (width * depth * channels + 7) // 8
    rows, previous, step = [], bytes(stride), max(1, depth * channels // 8)
    for y in range(height):
        kind, line = data[y * (stride + 1)], bytearray(data[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        if kind == 2:                                # "up": add the row above, byte by byte
            line = bytearray((a + b) & 255 for a, b in zip(line, previous))
        elif kind:
            for i in range(stride):                  # left, average and Paeth depend on the bytes just decoded
                left = line[i - step] if i >= step else 0
                up, corner = previous[i], (previous[i - step] if i >= step else 0)
                if kind == 1:
                    line[i] = (line[i] + left) & 255
                elif kind == 3:
                    line[i] = (line[i] + (left + up) // 2) & 255
                elif kind == 4:
                    guess = left + up - corner
                    near = min((abs(guess - left), 0, left), (abs(guess - up), 1, up),
                               (abs(guess - corner), 2, corner))
                    line[i] = (line[i] + near[2]) & 255
        previous = bytes(line)
        if channels == 3:
            rows.append(list(zip(line[0::3], line[1::3], line[2::3])))
        elif depth == 8:
            rows.append(list(line))
        else:
            top = (1 << depth) - 1
            values = [(line[(x * depth) // 8] >> (8 - depth - (x * depth) % 8)) & top for x in range(width)]
            rows.append([v * 255 // top for v in values])
    return rows


def read_bmp(path: str | Path) -> list[list[tuple[int, int, int]]]:
    """Rows of (R, G, B) from an uncompressed 24-bit or 8-bit palette BMP."""

    raw = Path(path).read_bytes()
    offset = struct.unpack("<I", raw[10:14])[0]
    header = struct.unpack("<I", raw[14:18])[0]
    width, height, _, bits = struct.unpack("<iiHH", raw[18:30])
    palette = []
    if bits == 8:
        colors = (offset - 14 - header) // 4
        palette = [tuple(raw[14 + header + 4 * i : 14 + header + 4 * i + 3][::-1]) for i in range(colors)]
    stride = (width * bits // 8 + 3) // 4 * 4               # rows are padded to 4 bytes
    rows = []
    for y in range(abs(height)):
        start = offset + y * stride
        if bits == 24:                                     # stored as B, G, R
            rows.append([(raw[start + 3 * x + 2], raw[start + 3 * x + 1], raw[start + 3 * x]) for x in range(width)])
        else:
            rows.append([palette[raw[start + x]] for x in range(width)])
    return rows[::-1] if height > 0 else rows               # positive height: bottom row first
