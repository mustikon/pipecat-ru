"""Streaming synthesis: the base-class plumbing and the Qwen3 fast path (no GPU)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.tests.utils import run_test

import pipecat_ru.tts.qwen3 as qwen3_module
from pipecat_ru.tts._base import RussianLocalTTSService, iterate_in_thread
from pipecat_ru.tts.qwen3 import Qwen3TTSService, supports_fast_path
from tests.conftest import FakeQwenModel
from tests.test_tts import llm_reply

SR = 24000


def tone(n: int) -> np.ndarray:
    return (0.1 * np.sin(np.arange(n) / 10.0)).astype(np.float32)


class FakeFastGenerator:
    """Stands in for Qwen3FastGenerator: runs the qwen-tts call, yields chunks."""

    instances: list[FakeFastGenerator] = []

    def __init__(self, tts: Any, *, max_seq_len: int = 2048) -> None:
        self.tts = tts
        self.max_seq_len = max_seq_len
        self.sample_rate = SR
        self.warmups = 0
        self.calls: list[dict[str, Any]] = []
        FakeFastGenerator.instances.append(self)

    def warmup(self, call: Callable[[], Any], **kwargs: Any) -> None:
        self.warmups += 1
        self.warmup_kwargs = kwargs

    def stream(
        self,
        call: Callable[[], Any],
        *,
        ref_codes: Any = None,
        first_chunk_frames: int = 3,
        chunk_frames: int = 8,
        max_frames: int | None = None,
    ) -> Iterator[np.ndarray]:
        call()  # records the text and voice on the fake model
        self.calls.append(
            {"ref_codes": ref_codes, "first": first_chunk_frames, "chunk": chunk_frames}
        )
        yield tone(first_chunk_frames * 1920)
        for _ in range(2):
            yield tone(chunk_frames * 1920)


@pytest.fixture
def fake_fast(monkeypatch: pytest.MonkeyPatch) -> type[FakeFastGenerator]:
    FakeFastGenerator.instances = []
    monkeypatch.setattr(qwen3_module, "Qwen3FastGenerator", FakeFastGenerator)
    return FakeFastGenerator


class PromptItem:
    def __init__(self, icl_mode: bool) -> None:
        self.icl_mode = icl_mode
        self.ref_code = "ref-codes"


class CloneModel(FakeQwenModel):
    def __init__(self, icl_mode: bool = True) -> None:
        super().__init__()
        self.icl_mode = icl_mode

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        self.prompts_built += 1
        return [PromptItem(self.icl_mode)]


async def test_fast_path_streams_chunks_with_clone_prompt(
    fake_fast: type[FakeFastGenerator],
) -> None:
    model = CloneModel()
    tts = Qwen3TTSService(
        model=model, ref_audio="me.wav", ref_text="Привет.", fast=True, first_chunk_frames=2
    )
    down, _ = await run_test(tts, frames_to_send=llm_reply("Хорошо, сейчас посмотрю."))
    gen = fake_fast.instances[0]
    assert gen.warmups == 1
    assert gen.warmup_kwargs == {
        "ref_codes": "ref-codes",
        "first_chunk_frames": 2,
        "chunk_frames": 8,
    }
    assert gen.calls[0] == {"ref_codes": "ref-codes", "first": 2, "chunk": 8}
    assert model.clone_calls[0]["text"] == "Хорошо, сейчас посмотрю."
    audio = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    assert len(audio) == 2 * (2 + 8 + 8) * 1920  # every streamed chunk reached the output


def test_fast_path_skips_ref_codes_without_icl(fake_fast: type[FakeFastGenerator]) -> None:
    tts = Qwen3TTSService(
        model=CloneModel(icl_mode=False), ref_audio="me.wav", x_vector_only=True, fast=True
    )
    tts.load_model()
    list(tts.synthesize_stream("Да."))
    assert fake_fast.instances[0].calls[0]["ref_codes"] is None


def test_fast_synthesize_concatenates_chunks(fake_fast: type[FakeFastGenerator]) -> None:
    model = FakeQwenModel()
    tts = Qwen3TTSService(model=model, speaker="Ryan", instruct="спокойно", fast=True)
    tts.load_model()
    audio, sr = tts.synthesize("Привет.")
    assert sr == SR
    assert len(audio) == (3 + 8 + 8) * 1920
    assert model.custom_calls[0]["speaker"] == "Ryan"
    assert model.custom_calls[0]["instruct"] == "спокойно"


def test_fast_path_is_off_for_test_doubles_by_default(fake_fast: type[FakeFastGenerator]) -> None:
    tts = Qwen3TTSService(model=FakeQwenModel(), speaker="Ryan")
    tts.load_model()
    assert not fake_fast.instances
    assert not supports_fast_path(FakeQwenModel())
    chunks = list(tts.synthesize_stream("Привет."))
    assert len(chunks) == 1


class StreamingService(RussianLocalTTSService):
    """Yields a few chunks; can fail mid-stream or wait for a release signal."""

    def __init__(self, fail_after: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail_after = fail_after
        self.closed = threading.Event()

    def load_model(self) -> None:
        pass

    def synthesize(self, text: str) -> tuple[Any, int]:
        raise AssertionError("not used")

    def synthesize_stream(self, text: str) -> Iterator[tuple[Any, int]]:
        try:
            for i in range(3):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("GPU fell over")
                yield tone(4800), SR
        finally:
            self.closed.set()


async def test_base_streams_every_chunk() -> None:
    tts = StreamingService(normalizer=None)
    down, _ = await run_test(tts, frames_to_send=llm_reply("Привет."))
    audio = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    assert len(audio) == 3 * 4800 * 2
    assert tts.closed.is_set()


async def test_base_reports_mid_stream_errors() -> None:
    tts = StreamingService(fail_after=1, normalizer=None)
    frames = [f async for f in tts.run_tts("Привет.", "ctx")]
    assert any(isinstance(f, ErrorFrame) and "GPU fell over" in f.error for f in frames)
    audio = b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame))
    assert len(audio) == 4800 * 2  # the chunk before the failure still played
    assert tts.closed.is_set()


async def test_iterate_in_thread_closes_abandoned_iterators() -> None:
    produced: list[int] = []
    closed = threading.Event()
    release = threading.Event()

    def gen() -> Iterator[int]:
        try:
            for i in range(100):
                produced.append(i)
                yield i
                release.wait(1.0)
        finally:
            closed.set()

    agen = iterate_in_thread(gen())
    assert await anext(agen) == 0
    await agen.aclose()  # the consumer was interrupted
    release.set()
    await asyncio.to_thread(closed.wait, 2.0)
    assert closed.is_set()
    assert len(produced) <= 3


async def test_iterate_in_thread_propagates_errors() -> None:
    def gen() -> Iterator[int]:
        yield 1
        raise ValueError("boom")

    agen = iterate_in_thread(gen())
    assert await anext(agen) == 1
    with pytest.raises(ValueError, match="boom"):
        await anext(agen)
