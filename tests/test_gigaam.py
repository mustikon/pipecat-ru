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
            {
                "source": "auto",
                "device": "cuda",
                "fp16_encoder": False,
                "use_flash": False,
                "download_root": None,
            },
        )
    ]


def test_service_properties() -> None:
    stt = GigaAMSTTService(model=FakeGigaAM())
    assert stt.wants_wav_segments is False
    assert stt.segment_sample_rate == 16000
    assert stt.language_to_service_language(Language.EN) == "ru"


def _fake_gigaam_module(monkeypatch: object, load_model: object) -> None:
    import sys
    import types

    module = types.ModuleType("gigaam")
    module.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaam", module)  # type: ignore[attr-defined]


def test_loader_falls_back_to_hugging_face_when_download_fails(monkeypatch: object) -> None:
    import pipecat_ru.stt.gigaam as mod

    def broken(name: str, **kwargs: object) -> object:
        raise RuntimeError("Download failed: certificate has expired")

    _fake_gigaam_module(monkeypatch, broken)
    seen: list[tuple[str, object, object]] = []

    def fake_hf(name: str, *, device: object, revision: object, **kwargs: object) -> str:
        seen.append((name, device, revision))
        return "hf-model"

    monkeypatch.setattr(mod, "_load_from_hf", fake_hf)  # type: ignore[attr-defined]
    assert mod.load_gigaam_model("v3_e2e_ctc", device="cuda") == "hf-model"
    assert seen == [("v3_e2e_ctc", "cuda", None)]


def test_loader_package_source_does_not_fall_back(monkeypatch: object) -> None:
    import pytest

    import pipecat_ru.stt.gigaam as mod

    def broken(name: str, **kwargs: object) -> object:
        raise RuntimeError("Download failed")

    _fake_gigaam_module(monkeypatch, broken)
    with pytest.raises(RuntimeError, match="Download failed"):
        mod.load_gigaam_model(source="package")


def test_loader_uses_package_when_it_works(monkeypatch: object) -> None:
    import pipecat_ru.stt.gigaam as mod

    calls: list[dict[str, object]] = []

    def ok(name: str, **kwargs: object) -> str:
        calls.append({"name": name, **kwargs})
        return "pkg-model"

    _fake_gigaam_module(monkeypatch, ok)
    assert mod.load_gigaam_model(device="cpu") == "pkg-model"
    assert calls == [
        {"name": "v3_e2e_rnnt", "fp16_encoder": True, "use_flash": False, "device": "cpu"}
    ]


def test_loader_rejects_unknown_source_and_unpinned_model() -> None:
    import pytest

    import pipecat_ru.stt.gigaam as mod

    with pytest.raises(ValueError, match="source"):
        mod.load_gigaam_model(source="cdn")
    with pytest.raises(ValueError, match="pinned"):
        mod._load_from_hf("v9_future", device="cpu", revision=None)


def test_hf_style_model_enables_tensor_path() -> None:
    class HFStyle(FakeGigaAM):
        _device = "cpu"
        head = object()
        decoding = object()

        def forward(self, *args: object) -> None:
            raise TypeError("not called in this test")

    assert GigaAMTranscriber(HFStyle()).fast_path is True


def test_retarget_maps_hf_config_onto_package_classes() -> None:
    import types

    from pipecat_ru.stt.gigaam import _retarget

    enc = types.ModuleType("gigaam.encoder")
    enc.ConformerEncoder = object  # type: ignore[attr-defined]
    dec = types.ModuleType("gigaam.decoding")
    dec.RNNTGreedyDecoding = object  # type: ignore[attr-defined]
    cfg = {
        "encoder": {"_target_": "modeling_gigaam.ConformerEncoder", "n_layers": 16},
        "decoding": {"_target_": "modeling_gigaam.RNNTGreedyDecoding"},
        "other": [{"_target_": "keep.me"}],
    }
    _retarget(cfg, {"gigaam.encoder": enc, "gigaam.decoding": dec})
    assert cfg["encoder"]["_target_"] == "gigaam.encoder.ConformerEncoder"
    assert cfg["decoding"]["_target_"] == "gigaam.decoding.RNNTGreedyDecoding"
    assert cfg["other"][0]["_target_"] == "keep.me"


def test_retarget_rejects_unknown_class() -> None:
    import pytest

    from pipecat_ru.stt.gigaam import _retarget

    with pytest.raises(ValueError, match="NewHead"):
        _retarget({"_target_": "modeling_gigaam.NewHead"}, {})
