from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_ru.turns import (
    BackchannelUserTurnStartStrategy,
    is_backchannel,
    russian_user_turn_strategies,
)


@pytest.mark.parametrize(
    "text",
    ["ага", "Угу.", "да-да", "ну да", "Ага, понятно!", "мгм", "так", "Хорошо.", "окей"],
)
def test_backchannels(text: str) -> None:
    assert is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    ["стоп", "подожди", "нет", "да, но погоди", "а сколько стоит?", "", "ага ага ага ага"],
)
def test_not_backchannels(text: str) -> None:
    assert not is_backchannel(text)


def test_custom_vocabulary() -> None:
    assert is_backchannel("лады", vocabulary={"лады"})
    assert not is_backchannel("ага", vocabulary={"лады"})


class Recorder:
    """Collects the strategy's events."""

    def __init__(self, strategy: BackchannelUserTurnStartStrategy) -> None:
        self.started = 0
        self.resets = 0
        strategy.add_event_handler("on_user_turn_started", self._started)
        strategy.add_event_handler("on_reset_aggregation", self._reset)

    async def _started(self, strategy: Any, params: Any) -> None:
        self.started += 1

    async def _reset(self, strategy: Any) -> None:
        self.resets += 1


def make(**kwargs: Any) -> tuple[BackchannelUserTurnStartStrategy, Recorder]:
    kwargs.setdefault("max_backchannel_secs", 0.05)
    kwargs.setdefault("vad_stop_secs", 0.0)
    strategy = BackchannelUserTurnStartStrategy(**kwargs)
    return strategy, Recorder(strategy)


def final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user", "2026-11-15T00:00:00")


def interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text, "user", "2026-11-15T00:00:00")


async def test_bot_silent_vad_starts_turn() -> None:
    s, rec = make()
    await s.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    assert rec.started == 1


async def test_bot_silent_transcript_fallback_starts_turn() -> None:
    s, rec = make()
    await s.process_frame(final("ага"))
    assert rec.started == 1


async def test_short_aga_while_bot_speaks_is_ignored() -> None:
    s, rec = make(max_backchannel_secs=0.5)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame(start_secs=0.0))
    await s.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.0))
    await s.process_frame(final("Ага."))
    await asyncio.sleep(0.6)  # the long-speech timer must have been cancelled
    assert rec.started == 0
    assert rec.resets == 1


async def test_short_stop_command_interrupts() -> None:
    s, rec = make(max_backchannel_secs=0.5)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await s.process_frame(final("Стоп!"))
    assert rec.started == 1
    assert rec.resets == 0


async def test_long_speech_interrupts_before_transcript() -> None:
    s, rec = make(max_backchannel_secs=0.05)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame(start_secs=0.0))
    await asyncio.sleep(0.2)
    assert rec.started == 1
    # the transcript of the same utterance must not start a second turn
    await s.handle_user_turn_started()
    await s.process_frame(final("ага, слушай, давай про другое"))
    assert rec.started == 2  # bot no longer speaking -> fallback path; controller dedups


async def test_vad_start_secs_counts_towards_duration() -> None:
    s, rec = make(max_backchannel_secs=0.2, vad_stop_secs=0.0)
    await s.process_frame(BotStartedSpeakingFrame())
    # VAD needed 0.2 s to confirm speech: the utterance is already long enough
    await s.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await asyncio.sleep(0.05)
    assert rec.started == 1


async def test_interim_non_backchannel_interrupts_immediately() -> None:
    s, rec = make(max_backchannel_secs=5.0)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(interim("подожди"))
    assert rec.started == 1
    await s.cleanup()


async def test_interim_backchannel_neither_interrupts_nor_resets() -> None:
    s, rec = make(max_backchannel_secs=5.0)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(interim("угу"))
    assert (rec.started, rec.resets) == (0, 0)
    await s.cleanup()


async def test_interims_ignored_when_disabled() -> None:
    s, rec = make(max_backchannel_secs=5.0, use_interim=False)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(interim("подожди"))
    assert rec.started == 0
    await s.cleanup()


async def test_backchannel_said_as_bot_finishes_is_still_dropped() -> None:
    s, rec = make(max_backchannel_secs=0.5)
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await s.process_frame(BotStoppedSpeakingFrame())
    await s.process_frame(final("угу"))
    assert (rec.started, rec.resets) == (0, 1)


async def test_after_bot_stops_speech_starts_turns_again() -> None:
    s, rec = make()
    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(BotStoppedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    assert rec.started == 1


async def test_custom_backchannels_are_normalized() -> None:
    s, _ = make(backchannels=["Ещё"])
    assert s.is_backchannel("еще")


def test_russian_user_turn_strategies_with_smart_turn() -> None:
    strategies = russian_user_turn_strategies()
    assert isinstance(strategies, UserTurnStrategies)
    assert strategies.start is not None and strategies.stop is not None
    assert isinstance(strategies.start[0], BackchannelUserTurnStartStrategy)
    analyzer = strategies.stop[0]._turn_analyzer  # type: ignore[attr-defined]
    assert type(analyzer).__name__ == "LocalSmartTurnAnalyzerV3"


def test_russian_user_turn_strategies_without_smart_turn() -> None:
    strategies = russian_user_turn_strategies(smart_turn=False, max_backchannel_secs=0.4)
    assert strategies.start is not None
    start = strategies.start[0]
    assert isinstance(start, BackchannelUserTurnStartStrategy)
    assert start.max_backchannel_secs == 0.4
