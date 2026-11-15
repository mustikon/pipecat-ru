"""Real-model tests for a CUDA machine (e.g. an RTX 5090 on Windows 11).

Skipped unless ``PIPECAT_RU_GPU_TESTS=1``. They download the models on first run
(GigaAM v3 ~1 GB, Qwen3-TTS 1.7B ~4 GB, Silero ~100 MB, ruaccent ~0.5 GB).

    uv sync --all-extras
    uv pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"
    $env:PIPECAT_RU_GPU_TESTS = "1"                     # PowerShell
    $env:PIPECAT_RU_REF_AUDIO = "C:\\voice\\me.wav"      # optional: your voice for cloning
    $env:PIPECAT_RU_REF_TEXT  = "Текст, который звучит в записи."
    uv run pytest -m gpu -s

Each test is skipped (not failed) when its optional package is missing.
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np
import pytest

from pipecat_ru.bench import edit_distance, normalize_for_wer, resample_linear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("PIPECAT_RU_GPU_TESTS") != "1",
        reason="set PIPECAT_RU_GPU_TESTS=1 to run real-model tests",
    ),
]

PHRASE = "Привет! Напомни мне завтра в девять утра позвонить маме."


def need(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed")


def need_cuda() -> None:
    need("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_for_wer(ref).split(), normalize_for_wer(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


@pytest.fixture(scope="module")
def gigaam_transcriber():  # type: ignore[no-untyped-def]
    need_cuda()
    need("gigaam")
    from pipecat_ru.stt.gigaam import GigaAMTranscriber, load_gigaam_model

    return GigaAMTranscriber(load_gigaam_model(device="cuda"))


@pytest.fixture(scope="module")
def silero_audio() -> tuple[np.ndarray, int]:
    need("torch")
    need("silero")
    from pipecat_ru.text import RussianTextNormalizer
    from pipecat_ru.tts.silero import SileroTTSService

    tts = SileroTTSService(silero_sample_rate=48000)
    tts.load_model()
    samples, sr = tts.synthesize(RussianTextNormalizer()(PHRASE))
    return np.asarray(samples.detach().cpu().numpy(), dtype=np.float32), sr


def test_gigaam_transcribes_silero_speech(gigaam_transcriber, silero_audio) -> None:  # type: ignore[no-untyped-def]
    samples, sr = silero_audio
    audio16 = resample_linear(samples, sr, 16000)
    gigaam_transcriber.transcribe(audio16)  # warm-up
    t0 = time.perf_counter()
    text = gigaam_transcriber.transcribe(audio16)
    latency = time.perf_counter() - t0
    print(f"\nGigaAM: {text!r} in {latency * 1000:.0f} ms")
    assert wer(PHRASE, text) <= 0.2
    assert latency < 1.0


def test_gigaam_tensor_path_matches_file_path(gigaam_transcriber, silero_audio) -> None:  # type: ignore[no-untyped-def]
    samples, sr = silero_audio
    audio16 = resample_linear(samples, sr, 16000)
    assert gigaam_transcriber.fast_path, "tensor fast path not available for this GigaAM"
    fast = gigaam_transcriber.transcribe(audio16)
    slow = gigaam_transcriber._transcribe_file(audio16)
    assert normalize_for_wer(fast) == normalize_for_wer(slow)


def test_gigaam_silence_gives_no_text(gigaam_transcriber) -> None:  # type: ignore[no-untyped-def]
    assert gigaam_transcriber.transcribe(np.zeros(16000 * 2, dtype=np.float32)) == ""


def test_qwen3_tts_roundtrip(gigaam_transcriber) -> None:  # type: ignore[no-untyped-def]
    need_cuda()
    need("qwen_tts")
    from pipecat_ru.text import RussianTextNormalizer, RussianTextSplitter
    from pipecat_ru.tts.qwen3 import Qwen3TTSService

    ref_audio = os.environ.get("PIPECAT_RU_REF_AUDIO")
    ref_text = os.environ.get("PIPECAT_RU_REF_TEXT")
    if ref_audio:
        tts = Qwen3TTSService(
            ref_audio=ref_audio, ref_text=ref_text, x_vector_only=ref_text is None
        )
    else:
        tts = Qwen3TTSService(speaker="Ryan")
    tts.load_model()
    normalizer = RussianTextNormalizer()
    first = RussianTextSplitter().split(PHRASE)[0]
    tts.synthesize(normalizer(first))  # warm-up
    t0 = time.perf_counter()
    tts.synthesize(normalizer(first))
    first_audio = time.perf_counter() - t0
    full, sr = tts.synthesize(normalizer(PHRASE))
    full = np.asarray(full, dtype=np.float32)
    text = gigaam_transcriber.transcribe(resample_linear(full, sr, 16000))
    print(f"\nQwen3-TTS first chunk {first!r}: {first_audio * 1000:.0f} ms; heard {text!r}")
    assert len(full) / sr > 1.0
    assert wer(PHRASE, text) <= 0.35


def test_qwen3_tts_fast_streaming(gigaam_transcriber) -> None:  # type: ignore[no-untyped-def]
    need_cuda()
    need("qwen_tts")
    from pipecat_ru.text import RussianTextNormalizer, RussianTextSplitter
    from pipecat_ru.tts.qwen3 import Qwen3TTSService

    tts = Qwen3TTSService(speaker="Ryan", fast=True)
    tts.load_model()  # also captures the CUDA graph
    normalizer = RussianTextNormalizer()
    first = normalizer(RussianTextSplitter().split(PHRASE)[0])
    t0 = time.perf_counter()
    chunks = []
    first_audio = None
    sr = 24000
    for samples, rate in tts.synthesize_stream(first):
        first_audio = first_audio or time.perf_counter() - t0
        chunks.append(samples)
        sr = rate
    assert first_audio is not None
    t0 = time.perf_counter()
    full = np.concatenate([c for c, _ in tts.synthesize_stream(normalizer(PHRASE))])
    rtf = (time.perf_counter() - t0) / (len(full) / sr)
    text = gigaam_transcriber.transcribe(resample_linear(full, sr, 16000))
    print(
        f"\nQwen3-TTS fast: first audio {first_audio * 1000:.0f} ms, RTF {rtf:.2f}; heard {text!r}"
    )
    assert first_audio < 0.5
    assert rtf < 1.0
    assert wer(PHRASE, text) <= 0.35


def test_ruaccent_stress_marks() -> None:
    need("ruaccent")
    from pipecat_ru.text import StressMarker

    marked = StressMarker("plus")("Мы подошли к старому замку на берегу реки.")
    print(f"\nruaccent: {marked}")
    assert "+" in marked
