"""Shared pieces of the embodied example: a tiny scene, a PNG codec and the message shapes.

The robot lives on a unit square. Its end effector and the target are drawn as
squares in a small image, the "camera frame" the policy looks at. Everything
is standard library so the example runs anywhere the SDK runs.
"""
from __future__ import annotations
import base64
import struct
import zlib

WIDTH = 64
HEIGHT = 64
BACKGROUND = (236, 238, 241)
TARGET = (220, 60, 50)
EFFECTOR = (40, 160, 90)
INSTRUCTION = "move the gripper onto the red marker"


def encode_png(pixels: bytearray, width: int, height: int) -> bytes:
    """RGB rows to a PNG with filter type 0 on every scanline."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """The width, height and RGB rows of a PNG written by ``encode_png``."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    offset, width, height, idat = 8, 0, 0, b""
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            if depth != 8 or color != 2:
                raise ValueError("only 8-bit RGB images are supported")
        elif kind == b"IDAT":
            idat += body
        offset += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 3
    pixels = bytearray()
    for y in range(height):
        row = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        if row[0] != 0:
            raise ValueError("only unfiltered scanlines are supported")
        pixels += row[1:]
    return width, height, bytes(pixels)


def render(effector: tuple[float, float], target: tuple[float, float]) -> bytes:
    """The camera frame: background, the target square and the effector square."""
    pixels = bytearray(BACKGROUND * (WIDTH * HEIGHT))
    for position, color in ((target, TARGET), (effector, EFFECTOR)):
        cx, cy = int(position[0] * (WIDTH - 1)), int(position[1] * (HEIGHT - 1))
        for y in range(max(0, cy - 3), min(HEIGHT, cy + 4)):
            for x in range(max(0, cx - 3), min(WIDTH, cx + 4)):
                pixels[(y * WIDTH + x) * 3:(y * WIDTH + x) * 3 + 3] = bytes(color)
    return encode_png(pixels, WIDTH, HEIGHT)


def locate(pixels: bytes, width: int, height: int, color: tuple[int, int, int], tolerance: int = 40) -> tuple[float, float] | None:
    """Centroid of the pixels close to ``color``, in unit coordinates."""
    total, sx, sy = 0, 0, 0
    for index in range(0, len(pixels), 3):
        if all(abs(pixels[index + i] - color[i]) <= tolerance for i in range(3)):
            pixel = index // 3
            sx += pixel % width
            sy += pixel // width
            total += 1
    if total == 0:
        return None
    return sx / total / (width - 1), sy / total / (height - 1)


def observation(effector: tuple[float, float], target: tuple[float, float]) -> dict:
    """What the robot sends per control step: an instruction, its state and the camera frame."""
    return {
        "instruction": INSTRUCTION,
        "state": [effector[0], effector[1]],
        "image_png": base64.b64encode(render(effector, target)).decode("ascii"),
    }
