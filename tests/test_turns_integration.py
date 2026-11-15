"""The strategy inside Pipecat's real LLMUserAggregator (no models involved)."""

from __future__ import annotations

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_ru.turns import russian_user_turn_strategies


async def run_utterance(text: str) -> tuple[list[str], LLMContext]:
    context = LLMContext()
    aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=russian_user_turn_strategies(
                smart_turn=False, max_backchannel_secs=0.5
            )
        ),
    )
    frames = [
        BotStartedSpeakingFrame(),
        VADUserStartedSpeakingFrame(start_secs=0.2),
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
        TranscriptionFrame(text, "user", "2026-11-15T00:00:00"),
        SleepFrame(sleep=1.0),
    ]
    down, _ = await run_test(aggregator, frames_to_send=frames)
    return [type(f).__name__ for f in down], context


async def test_backchannel_does_not_interrupt_or_reach_the_llm() -> None:
    names, context = await run_utterance("ага")
    assert UserStartedSpeakingFrame.__name__ not in names
    assert InterruptionFrame.__name__ not in names
    assert LLMContextFrame.__name__ not in names
    assert not any("ага" in str(m) for m in context.get_messages())


async def test_short_command_interrupts_and_reaches_the_llm() -> None:
    names, context = await run_utterance("стоп")
    assert InterruptionFrame.__name__ in names
    assert LLMContextFrame.__name__ in names
    assert any("стоп" in str(m) for m in context.get_messages())
