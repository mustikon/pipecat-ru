"""Turn-taking helpers for Russian conversations."""

from __future__ import annotations

from typing import Any

from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_ru.turns.backchannel import (
    DEFAULT_BACKCHANNELS,
    BackchannelUserTurnStartStrategy,
    is_backchannel,
)


def russian_user_turn_strategies(
    *,
    max_backchannel_secs: float = 0.6,
    vad_stop_secs: float = 0.2,
    smart_turn: bool = True,
    **start_kwargs: Any,
) -> UserTurnStrategies:
    """User turn strategies tuned for Russian.

    Start: :class:`BackchannelUserTurnStartStrategy` (short "ага"/"угу" while the bot
    speaks do not interrupt it). Stop: smart-turn v3.2, which supports Russian
    (``LocalSmartTurnAnalyzerV3``, bundled with pipecat-ai and run on the CPU), or
    Pipecat's default stop strategy when ``smart_turn`` is False.

    Args:
        max_backchannel_secs: See :class:`BackchannelUserTurnStartStrategy`.
        vad_stop_secs: Your VAD's ``stop_secs``; smart-turn works best with 0.2.
        smart_turn: Use smart-turn v3.2 as the stop strategy.
        **start_kwargs: More options for :class:`BackchannelUserTurnStartStrategy`.
    """
    start = [
        BackchannelUserTurnStartStrategy(
            max_backchannel_secs=max_backchannel_secs,
            vad_stop_secs=vad_stop_secs,
            **start_kwargs,
        )
    ]
    if not smart_turn:
        return UserTurnStrategies(start=list(start))

    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
        TurnAnalyzerUserTurnStopStrategy,
    )

    return UserTurnStrategies(
        start=list(start),
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
    )


__all__ = [
    "DEFAULT_BACKCHANNELS",
    "BackchannelUserTurnStartStrategy",
    "is_backchannel",
    "russian_user_turn_strategies",
]
