"""Shared base for the local Russian TTS services."""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, StartFrame
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.text.transforms import strip_markdown
from pipecat.utils.tracing.service_decorators import traced_tts

from pipecat_ru._audio import chunk_bytes, float_to_pcm16
from pipecat_ru.text.normalizer import RussianTextNormalizer
from pipecat_ru.text.splitter import RussianTextAggregator, SplitterConfig

# Audio is handed to Pipecat in pieces of this length, so playback can start
# before a long chunk has been copied through the pipeline.
_PUSH_SECS = 0.2


class _Default:
    """Sentinel: use the service's default normalizer/splitter."""


DEFAULT: Any = _Default()


class RussianLocalTTSService(TTSService):
    """Base for local TTS models that synthesize a whole text chunk per call.

    Adds two Russian-specific defaults on top of Pipecat's ``TTSService``:

    * :class:`~pipecat_ru.text.RussianTextAggregator` replaces the sentence
      aggregator, so the first chunk of each reply is a short clause and the first
      audio comes out sooner. Pass ``splitter=None`` to keep Pipecat's aggregator,
      or a :class:`~pipecat_ru.text.SplitterConfig` to tune it.
    * Markdown stripping and :class:`~pipecat_ru.text.RussianTextNormalizer` run as
      text transforms. Pass ``normalizer=None`` to switch normalization off, or
      your own normalizer (custom lexicon, stress marks).

    Subclasses implement :meth:`load_model` and :meth:`synthesize`.
    """

    def __init__(
        self,
        *,
        normalizer: RussianTextNormalizer | None = DEFAULT,
        splitter: SplitterConfig | None = DEFAULT,
        text_transforms: list[tuple[TextType | str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Set up text handling; see the class docstring."""
        if normalizer is DEFAULT:
            normalizer = self.default_normalizer()
        self.normalizer: RussianTextNormalizer | None = normalizer
        transforms: list[tuple[TextType | str, Any]] = [("*", strip_markdown)]
        if normalizer is not None:
            transforms.append(("*", normalizer.transform))
        transforms.extend(text_transforms or [])

        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        super().__init__(text_transforms=transforms, **kwargs)

        if splitter is not None and self._text_aggregation_mode == TextAggregationMode.SENTENCE:
            config = None if splitter is DEFAULT else splitter
            # TTSService has no public hook for a custom aggregator; it declares the
            # attribute as SimpleTextAggregator but only uses the base interface.
            self._text_aggregator = RussianTextAggregator(config)  # type: ignore[assignment]

        self._model_lock = asyncio.Lock()
        self._loaded = False

    # ------------------------------------------------------------------ subclass API

    def default_normalizer(self) -> RussianTextNormalizer | None:
        """The normalizer used when none is passed."""
        return RussianTextNormalizer()

    @abstractmethod
    def load_model(self) -> None:
        """Load the model (runs in a worker thread). Must be idempotent."""

    @abstractmethod
    def synthesize(self, text: str) -> tuple[Any, int]:
        """Synthesize ``text`` (runs in a worker thread): (float samples, sample rate)."""

    # ------------------------------------------------------------------ Pipecat API

    def can_generate_metrics(self) -> bool:
        """This service reports TTFB and usage metrics."""
        return True

    async def start(self, frame: StartFrame) -> None:
        """Load the model before the first reply."""
        await super().start(frame)
        await self.ensure_loaded()

    async def ensure_loaded(self) -> None:
        """Load the model now if needed; safe to call more than once."""
        async with self._model_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            await asyncio.to_thread(self.load_model)
            self._loaded = True
            logger.info(f"{self}: model ready in {time.perf_counter() - t0:.1f}s")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize one chunk of text and stream it out as audio frames."""
        try:
            await self.ensure_loaded()
            await self.start_tts_usage_metrics(text)
            samples, sample_rate = await asyncio.to_thread(self.synthesize, text)
            pcm = float_to_pcm16(samples)
            if not pcm:
                logger.warning(f"{self}: no audio for [{text}]")
                return
            piece = max(2, int(sample_rate * _PUSH_SECS) * 2)

            async def pieces() -> AsyncIterator[bytes]:
                for chunk in chunk_bytes(pcm, piece):
                    yield chunk

            async for frame in self._stream_audio_frames_from_iterator(
                pieces(), in_sample_rate=sample_rate, context_id=context_id
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.exception(f"{self}: synthesis failed for [{text}]")
            yield ErrorFrame(error=f"{type(self).__name__} synthesis failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
