from __future__ import annotations

import numpy as np
import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from pipecat_ru.text import RussianTextAggregator, RussianTextNormalizer, SplitterConfig
from pipecat_ru.tts.qwen3 import Qwen3TTSService
from pipecat_ru.tts.silero import SileroTTSService
from tests.conftest import FakeQwenModel, FakeSileroModel


def llm_reply(text: str, step: int = 4) -> list[object]:
    tokens = [text[i : i + step] for i in range(0, len(text), step)]
    return [
        LLMFullResponseStartFrame(),
        *[LLMTextFrame(t) for t in tokens],
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.3),
    ]


def clone_service(model: FakeQwenModel, **kwargs: object) -> Qwen3TTSService:
    return Qwen3TTSService(
        model=model, ref_audio="me.wav", ref_text="Привет, это мой голос.", **kwargs
    )


async def test_qwen3_speaks_short_first_chunk_then_sentences(fake_qwen: FakeQwenModel) -> None:
    tts = clone_service(fake_qwen)
    down, _ = await run_test(
        tts,
        frames_to_send=llm_reply("Хорошо, сейчас посмотрю, что будет 15 марта. Будет **ясно**!"),
    )
    texts = [c["text"] for c in fake_qwen.clone_calls]
    assert texts[0] == "Хорошо, сейчас посмотрю,"
    assert texts[1].startswith("что будет пятнадцат")
    assert texts[2] == "Будет ясно!"  # markdown stripped
    assert all(c["language"] == "Russian" for c in fake_qwen.clone_calls)
    assert all(c["voice_clone_prompt"] == ["prompt"] for c in fake_qwen.clone_calls)
    assert fake_qwen.prompts_built == 1
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)


async def test_qwen3_custom_voice_mode(fake_qwen: FakeQwenModel) -> None:
    tts = Qwen3TTSService(model=fake_qwen, speaker="Ryan", instruct="спокойно")
    await run_test(tts, frames_to_send=llm_reply("Привет, как у тебя дела сегодня?"))
    assert fake_qwen.custom_calls[0]["speaker"] == "Ryan"
    assert fake_qwen.custom_calls[0]["instruct"] == "спокойно"
    assert fake_qwen.prompts_built == 0


def test_qwen3_argument_validation() -> None:
    with pytest.raises(ValueError, match="ref_audio"):
        Qwen3TTSService(model=FakeQwenModel())
    with pytest.raises(ValueError, match="ref_text"):
        Qwen3TTSService(model=FakeQwenModel(), ref_audio="me.wav")
    Qwen3TTSService(model=FakeQwenModel(), ref_audio="me.wav", x_vector_only=True)


def test_qwen3_default_model_names() -> None:
    clone = clone_service(FakeQwenModel())
    assert clone._model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    preset = Qwen3TTSService(model=FakeQwenModel(), speaker="Ryan")
    assert preset._model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def test_russian_aggregator_is_installed_by_default() -> None:
    assert isinstance(clone_service(FakeQwenModel())._text_aggregator, RussianTextAggregator)


def test_splitter_can_be_disabled_or_tuned() -> None:
    off = clone_service(FakeQwenModel(), splitter=None)
    assert isinstance(off._text_aggregator, SimpleTextAggregator)
    tuned = clone_service(FakeQwenModel(), splitter=SplitterConfig(first_chunk_min_words=1))
    agg = tuned._text_aggregator
    assert isinstance(agg, RussianTextAggregator)
    assert agg._splitter.config.first_chunk_min_words == 1


def test_token_mode_keeps_pipecat_aggregator() -> None:
    tts = clone_service(FakeQwenModel(), text_aggregation_mode=TextAggregationMode.TOKEN)
    assert not isinstance(tts._text_aggregator, RussianTextAggregator)


async def test_normalizer_can_be_disabled(fake_qwen: FakeQwenModel) -> None:
    tts = clone_service(fake_qwen, normalizer=None)
    await run_test(tts, frames_to_send=llm_reply("Мне 21 год, а тебе сколько лет?"))
    assert fake_qwen.clone_calls[0]["text"] == "Мне 21 год,"


async def test_custom_normalizer_is_used(fake_qwen: FakeQwenModel) -> None:
    tts = clone_service(fake_qwen, normalizer=RussianTextNormalizer(lexicon={"Мустик": "Муст+ик"}))
    await run_test(tts, frames_to_send=llm_reply("Привет, Мустик, как дела?"))
    assert "Муст+ик" in fake_qwen.clone_calls[0]["text"]


async def test_synthesis_errors_become_error_frames() -> None:
    class Broken(FakeQwenModel):
        def generate_voice_clone(self, text: str, **kwargs: object) -> tuple[list[np.ndarray], int]:
            raise RuntimeError("boom")

    tts = clone_service(Broken())
    frames = [f async for f in tts.run_tts("Привет.", "ctx")]
    assert any(isinstance(f, ErrorFrame) and "boom" in f.error for f in frames)


async def test_silero_synthesizes_with_accent_args() -> None:
    model = FakeSileroModel()
    tts = SileroTTSService(model=model, speaker="baya", silero_sample_rate=24000)
    down, _ = await run_test(tts, frames_to_send=llm_reply("Привет, это запасной голос. Всё ок."))
    assert model.calls[0]["speaker"] == "baya"
    assert model.calls[0]["sample_rate"] == 24000
    assert model.calls[0]["put_accent"] is True
    assert model.calls[0]["text"] == "Привет, это запасной голос."
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)


def test_silero_drops_unsupported_args_once() -> None:
    model = FakeSileroModel(accept_extra_args=False)
    tts = SileroTTSService(model=model)
    tts.load_model()
    _, sr = tts.synthesize("Привет.")
    tts.synthesize("Ещё раз.")
    assert sr == 48000
    assert len(model.calls) == 2
    assert all("put_accent" not in c for c in model.calls)


def test_silero_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="silero_sample_rate"):
        SileroTTSService(model=FakeSileroModel(), silero_sample_rate=16000)
