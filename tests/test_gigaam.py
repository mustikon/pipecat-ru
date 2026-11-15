from __future__ import annotations

import os

import numpy as np
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from pipecat_ru.stt.gigaam import GIGAAM_SAMPLE_RATE, GigaAMSTTService, GigaAMTranscriber
from tests.conftest import FakeGigaAM


def tone(secs: float, sr: int = GIGAAM_SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(secs * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_transcriber_uses_public_api_and_cleans_up(fake_gigaam: FakeGigaAM) -> None:
    tr = GigaAMTranscriber(fake_gigaam)
    assert tr.fast_path is False
    assert tr.transcribe(tone(1.0)) == "Привет, как дела?"
    assert fake_gigaam.durations == [1.0]
    assert not os.path.exists(fake_gigaam.paths[0])


def test_transcriber_accepts_plain_string_results() -> None:
    assert GigaAMTranscriber(FakeGigaAM("да", as_str=True)).transcribe(tone(0.5)) == "да"


def test_long_audio_is_split_below_25_seconds() -> None:
    fake = FakeGigaAM("кусок")
    text = GigaAMTranscriber(fake, max_segment_secs=10.0).transcribe(tone(25.0))
    assert len(fake.durations) >= 3
    assert all(d <= 10.0 for d in fake.durations)
    assert abs(sum(fake.durations) - 25.0) < 0.01
    assert text == " ".join(["кусок"] * len(fake.durations))


def test_fast_path_falls_back_to_file_when_internals_differ() -> None:
    class Odd(FakeGigaAM):
        _device = "cpu"
        _dtype = None

        def forward(self, *args: object) -> None:
            raise TypeError("different signature")

        def _decode(self, *args: object) -> None:
            raise AssertionError("not reached")

    model = Odd("готово")
    tr = GigaAMTranscriber(model)
    assert tr.fast_path is True
    assert tr.transcribe(tone(0.3)) == "готово"  # torch missing or TypeError -> file path
    assert tr.fast_path is False


async def test_service_emits_russian_transcription(fake_gigaam: FakeGigaAM) -> None:
    stt = GigaAMSTTService(model=fake_gigaam, trailing_silence_secs=0.0)
    chunk = (tone(0.02) * 32767).astype("<i2").tobytes()
    frames = [
        VADUserStartedSpeakingFrame(),
        *[InputAudioRawFrame(chunk, GIGAAM_SAMPLE_RATE, 1) for _ in range(40)],
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.5),
    ]
    down, _ = await run_test(stt, frames_to_send=frames)
    transcripts = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [t.text for t in transcripts] == ["Привет, как дела?"]
    assert transcripts[0].language == Language.RU
    assert fake_gigaam.durations and 0.5 < fake_gigaam.durations[0] < 1.5


async def test_service_reports_model_errors() -> None:
    class Broken(FakeGigaAM):
        def transcribe(self, path: str) -> object:
            raise RuntimeError("CUDA out of memory")

    stt = GigaAMSTTService(model=Broken())
    frames = [f async for f in stt.run_stt((tone(0.2) * 32767).astype("<i2").tobytes())]
    assert len(frames) == 1 and isinstance(frames[0], ErrorFrame)
    assert "CUDA out of memory" in frames[0].error


async def test_service_skips_empty_transcripts() -> None:
    stt = GigaAMSTTService(model=FakeGigaAM(""))
    frames = [f async for f in stt.run_stt((tone(0.2) * 32767).astype("<i2").tobytes())]
    assert frames == []


async def test_service_loads_model_lazily(monkeypatch: object) -> None:
    import pipecat_ru.stt.gigaam as mod

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_load(name: str, **kwargs: object) -> FakeGigaAM:
        calls.append((name, kwargs))
        return FakeGigaAM("ок")

    monkeypatch.setattr(mod, "load_gigaam_model", fake_load)  # type: ignore[attr-defined]
    stt = GigaAMSTTService(device="cuda", fp16_encoder=False)
    await stt.ensure_loaded()
    await stt.ensure_loaded()
    assert calls == [
        (
            "v3_e2e_rnnt",
            {"device": "cuda", "fp16_encoder": False, "use_flash": False, "download_root": None},
        )
    ]


def test_service_properties() -> None:
    stt = GigaAMSTTService(model=FakeGigaAM())
    assert stt.wants_wav_segments is False
    assert stt.segment_sample_rate == 16000
    assert stt.language_to_service_language(Language.EN) == "ru"
