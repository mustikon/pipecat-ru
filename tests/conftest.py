"""Test doubles for the models: no GPU, no downloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class FakeAccentizer:
    """Stands in for ruaccent.RUAccent: stresses a few known words."""

    STRESS = {"замок": "з+амок", "молоко": "молок+о", "привет": "прив+ет"}

    def __init__(self) -> None:
        self.calls: list[str] = []

    def process_all(self, text: str) -> str:
        self.calls.append(text)
        out = text
        for plain, marked in self.STRESS.items():
            out = out.replace(plain, marked)
        return out


class FakeQwenModel:
    """Mimics qwen_tts.Qwen3TTSModel's generation API."""

    sample_rate = 24000

    def __init__(self) -> None:
        self.prompts_built = 0
        self.clone_calls: list[dict[str, Any]] = []
        self.custom_calls: list[dict[str, Any]] = []

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[str]:
        self.prompts_built += 1
        return ["prompt"]

    def _audio(self, text: str) -> np.ndarray:
        # 10 ms of a quiet tone per character, so longer text -> longer audio
        n = max(1, len(text)) * self.sample_rate // 100
        return (0.1 * np.sin(np.arange(n) / 10.0)).astype(np.float32)

    def generate_voice_clone(self, text: str, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.clone_calls.append({"text": text, **kwargs})
        return [self._audio(text)], self.sample_rate

    def generate_custom_voice(self, text: str, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.custom_calls.append({"text": text, **kwargs})
        return [self._audio(text)], self.sample_rate


class FakeTensor:
    """Duck-types the torch.Tensor methods the services use."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._data


class FakeSileroModel:
    """Mimics a Silero TTS model; optionally rejects put_accent/put_yo."""

    def __init__(self, accept_extra_args: bool = True) -> None:
        self.accept_extra_args = accept_extra_args
        self.calls: list[dict[str, Any]] = []

    def apply_tts(self, **kwargs: Any) -> FakeTensor:
        if not self.accept_extra_args and ("put_accent" in kwargs or "put_yo" in kwargs):
            raise TypeError("unexpected keyword argument 'put_accent'")
        self.calls.append(kwargs)
        n = kwargs["sample_rate"] // 10
        return FakeTensor(np.full(n, 0.05, dtype=np.float32))


class FakeGigaAM:
    """Mimics GigaAM's public transcribe(path) API (no tensor fast path)."""

    def __init__(self, text: str = "Привет, как дела?", as_str: bool = False) -> None:
        self.text = text
        self.as_str = as_str
        self.paths: list[str] = []
        self.durations: list[float] = []

    def transcribe(self, path: str) -> Any:
        import soundfile as sf

        data, sr = sf.read(path)
        self.paths.append(path)
        self.durations.append(len(data) / sr)
        if self.as_str:
            return self.text

        class Result:
            text = self.text

        return Result()


@pytest.fixture
def fake_accentizer() -> FakeAccentizer:
    return FakeAccentizer()


@pytest.fixture
def fake_qwen() -> FakeQwenModel:
    return FakeQwenModel()


@pytest.fixture
def fake_gigaam() -> FakeGigaAM:
    return FakeGigaAM()
