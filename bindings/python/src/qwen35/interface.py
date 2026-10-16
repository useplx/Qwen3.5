from __future__ import annotations

import struct
from dataclasses import dataclass

_INPUT_MAGIC = 0x51333549
_TEXT = 0
_IMAGE = 1
_VIDEO = 2


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class Image:
    encoded: bytes


@dataclass(frozen=True)
class Video:
    frames: tuple[Image, ...]
    frames_per_second: float = 2.0


Content = Text | Image | Video


def _field(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


@dataclass(frozen=True)
class Input:
    content: tuple[Content, ...]

    @classmethod
    def text(cls, value: str) -> Input:
        return cls((Text(value),))

    def encode(self) -> bytes:
        payload = bytearray(struct.pack("<II", _INPUT_MAGIC, len(self.content)))
        for item in self.content:
            if isinstance(item, Text):
                kind, value = _TEXT, item.value.encode("utf-8")
            elif isinstance(item, Image):
                kind, value = _IMAGE, item.encoded
            elif isinstance(item, Video):
                video = bytearray(struct.pack("<I", len(item.frames)))
                for frame in item.frames:
                    video.extend(_field(frame.encoded))
                video.extend(struct.pack("<f", item.frames_per_second))
                kind, value = _VIDEO, bytes(video)
            else:
                raise TypeError(f"unsupported Qwen3.5 content: {type(item).__name__}")
            payload.extend(struct.pack("<B", kind))
            payload.extend(_field(value))
        return bytes(payload)
