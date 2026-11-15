"""Backchannel-aware user turn start strategy for Russian.

While the bot speaks, a listener says "ага", "угу", "да", "так" to show they are
following. With Pipecat's default start strategies (VAD + transcription), each of
those interrupts the bot mid-sentence. This strategy replaces both defaults:

* Bot silent: a turn starts on VAD speech start, exactly like
  ``VADUserTurnStartStrategy`` (plus the transcription fallback).
* Bot speaking: an utterance interrupts only if it lasts longer than
  ``max_backchannel_secs`` (default 0.6 s) **or** its transcript contains anything
  other than backchannel words. A short "ага" is dropped from the user aggregation
  and the bot keeps talking; a short "стоп" or "подожди" still interrupts.

Pair it with smart-turn v3.2 (which understands Russian) as the stop strategy::

    from pipecat_ru.turns import russian_user_turn_strategies

    user_aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(user_turn_strategies=russian_user_turn_strategies()),
    )
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Iterable
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

# Lowercase, "ё" folded to "е". Multi-word fillers ("ну да") are covered because
# every word is checked on its own.
DEFAULT_BACKCHANNELS: frozenset[str] = frozenset(
    {
        "ага",
        "угу",
        "угум",
        "ог",
        "мгм",
        "мм",
        "ммм",
        "м",
        "хм",
        "хмм",
        "э",
        "эм",
        "да",
        "даа",
        "ну",
        "так",
        "вот",
        "ок",
        "окей",
        "ладно",
        "ясно",
        "понятно",
        "понял",
        "поняла",
        "хорошо",
        "верно",
        "точно",
        "именно",
        "конечно",
        "ого",
        "ой",
        "ах",
        "о",
        "ух",
        "угу-угу",
        "ага-ага",
        "да-да",
        "так-так",
        "ну-ну",
        "слушаю",
        "интересно",
        "правда",
        "ну-да",
        "ясненько",
        "aha",
        "ok",
        "okay",
        "mhm",
        "uh-huh",
        "yeah",
        "yes",
    }
)

_WORD_RE = re.compile(r"[\w]+(?:-[\w]+)*", re.UNICODE)


def backchannel_words(text: str) -> list[str]:
    """Split a transcript into lowercase words with "ё" folded to "е"."""
    return [w.replace("ё", "е") for w in _WORD_RE.findall(text.lower())]


def is_backchannel(
    text: str,
    *,
    vocabulary: Iterable[str] = DEFAULT_BACKCHANNELS,
    max_words: int = 3,
) -> bool:
    """True if ``text`` is only listener feedback ("ага", "угу, да", "ну да").

    Empty text is not a backchannel (there is nothing to classify).
    """
    words = backchannel_words(text)
    if not words or len(words) > max_words:
        return False
    vocab = vocabulary if isinstance(vocabulary, frozenset | set) else frozenset(vocabulary)
    return all(w in vocab for w in words)


class BackchannelUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Start user turns on speech, but let short backchannels pass while the bot talks.

    Args:
        max_backchannel_secs: Speech shorter than this, said while the bot is
            speaking, is a candidate backchannel; longer speech interrupts.
        vad_stop_secs: The ``stop_secs`` of your VAD (``VADParams.stop_secs``).
            VAD confirms the end of speech only after this much silence, so an
            utterance is declared long only after ``max_backchannel_secs +
            vad_stop_secs``.
        backchannels: Words that count as listener feedback (lowercase; "ё" as "е").
        max_backchannel_words: Transcripts with more words always interrupt.
        use_interim: React to interim transcripts too (useful with a streaming STT:
            a non-backchannel word interrupts before the utterance ends).
        **kwargs: Passed to ``BaseUserTurnStartStrategy`` (e.g. ``enable_interruptions``).
    """

    def __init__(
        self,
        *,
        max_backchannel_secs: float = 0.6,
        vad_stop_secs: float = 0.2,
        backchannels: Iterable[str] | None = None,
        max_backchannel_words: int = 3,
        use_interim: bool = True,
        **kwargs: Any,
    ) -> None:
        """Configure the thresholds; see the class docstring."""
        super().__init__(**kwargs)
        self.max_backchannel_secs = max_backchannel_secs
        self.vad_stop_secs = vad_stop_secs
        self.backchannels: frozenset[str] = frozenset(
            w.lower().replace("ё", "е") for w in (backchannels or DEFAULT_BACKCHANNELS)
        )
        self.max_backchannel_words = max_backchannel_words
        self.use_interim = use_interim

        self._bot_speaking = False
        # The utterance being judged: it started while the bot was speaking.
        self._judging = False
        self._user_speaking = False
        self._timer: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ Pipecat API

    async def handle_user_turn_started(self) -> None:
        """A turn started (by us or another strategy): stop judging the utterance."""
        await self._cancel_timer()
        self._judging = False
        # The turn start interrupts the bot, so it is no longer speaking.
        self._bot_speaking = False

    async def process_frame(self, frame: Frame) -> None:
        """Watch bot speech, VAD and transcripts to decide when the user's turn starts."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._on_vad_started(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._on_vad_stopped(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._on_transcript(frame.text, final=True)
        elif isinstance(frame, InterimTranscriptionFrame) and self.use_interim:
            await self._on_transcript(frame.text, final=False)

    async def cleanup(self) -> None:
        """Cancel the pending timer."""
        await self._cancel_timer()
        await super().cleanup()  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------ internals

    def is_backchannel(self, text: str) -> bool:
        """Classify a transcript with this strategy's vocabulary."""
        return is_backchannel(
            text, vocabulary=self.backchannels, max_words=self.max_backchannel_words
        )

    async def _on_vad_started(self, frame: VADUserStartedSpeakingFrame) -> None:
        self._user_speaking = True
        if not self._bot_speaking:
            await self.report_user_turn_started()
            return
        # VAD fires after start_secs of speech; count that time as already spoken.
        self._judging = True
        wait = max(0.0, self.max_backchannel_secs + self.vad_stop_secs - frame.start_secs)
        await self._cancel_timer()
        self._timer = self._create_task(self._long_speech_timer(wait))

    async def _on_vad_stopped(self, frame: VADUserStoppedSpeakingFrame) -> None:
        self._user_speaking = False
        await self._cancel_timer()

    async def _long_speech_timer(self, wait: float) -> None:
        await asyncio.sleep(wait)
        self._timer = None
        if self._judging and self._user_speaking:
            logger.debug(f"{self}: speech longer than {self.max_backchannel_secs}s, interrupting")
            self._judging = False
            await self.report_user_turn_started()

    async def _on_transcript(self, text: str, *, final: bool) -> None:
        if not text.strip():
            return
        if not (self._judging or self._bot_speaking):
            # Bot silent and no utterance under review: transcription fallback.
            await self.report_user_turn_started()
            return
        if self.is_backchannel(text):
            if final:
                logger.debug(f"{self}: backchannel {text!r} while the bot speaks, ignoring")
                self._judging = False
                await self.trigger_reset_aggregation()  # type: ignore[no-untyped-call]
            return
        logger.debug(f"{self}: {text!r} is not a backchannel, interrupting")
        await self._cancel_timer()
        self._judging = False
        await self.report_user_turn_started()

    def _create_task(self, coro: Any) -> asyncio.Task[None]:
        if self._task_manager is not None:
            task: asyncio.Task[None] = self._task_manager.create_task(
                coro, f"{self.name}::long_speech_timer"
            )
            return task
        return asyncio.create_task(coro)

    async def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is None or timer.done() or timer is asyncio.current_task():
            return
        if self._task_manager is not None:
            await self._task_manager.cancel_task(timer)
        else:
            timer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timer
