"""Small audio helpers shared by the services (numpy only, no torch)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def pcm16_to_float(audio: bytes) -> FloatArray:
    """Signed 16-bit little-endian PCM -> float32 in [-1, 1)."""
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: Any) -> bytes:
    """Float samples in [-1, 1] (numpy array, list or torch tensor) -> 16-bit PCM."""
    if hasattr(samples, "detach"):  # torch.Tensor without importing torch
        samples = samples.detach().float().cpu().numpy()
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def chunk_bytes(data: bytes, chunk_size: int) -> Iterator[bytes]:
    """Split ``data`` into pieces of at most ``chunk_size`` bytes."""
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


def split_long(samples: FloatArray, sample_rate: int, max_secs: float) -> list[FloatArray]:
    """Cut audio longer than ``max_secs`` into pieces, preferring quiet points.

    Each cut is placed at the quietest 20 ms window in the last third of the piece,
    so words are rarely split in half.
    """
    max_len = int(max_secs * sample_rate)
    if len(samples) <= max_len:
        return [samples]
    pieces: list[FloatArray] = []
    win = max(1, sample_rate // 50)
    start = 0
    while len(samples) - start > max_len:
        lo = start + (2 * max_len) // 3
        hi = start + max_len
        region = samples[lo:hi]
        n_win = max(1, len(region) // win)
        energy = np.square(region[: n_win * win]).reshape(n_win, win).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        pieces.append(samples[start:cut])
        start = cut
    pieces.append(samples[start:])
    return pieces
