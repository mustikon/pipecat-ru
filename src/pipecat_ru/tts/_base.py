"""Shared base for the local Russian TTS services."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, TypeVar

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


_T = TypeVar("_T")
_WORKERS: set[asyncio.Future[None]] = set()


async def iterate_in_thread(iterator: Iterator[_T]) -> AsyncGenerator[_T, None]:
    """Run a blocking iterator in a worker thread and yield its items.

    When the consumer stops early (an interruption cancels it), the worker stops
    after the item it is producing and closes the iterator in its own thread, so a
    generator's ``finally``/``with`` blocks run there.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    stop = threading.Event()

    def put(kind: str, value: Any) -> None:
        with contextlib.suppress(RuntimeError):  # the event loop is already closed
            loop.call_soon_threadsafe(queue.put_nowait, (kind, value))

    def work() -> None:
        try:
            for item in iterator:
                if stop.is_set():
                    break
                put("item", item)
        except BaseException as e:
            put("error", e)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            put("done", None)

    worker = asyncio.ensure_future(asyncio.to_thread(work))
    _WORKERS.add(worker)
    worker.add_done_callback(_WORKERS.discard)
    try:
        while True:
            kind, value = await queue.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        stop.set()


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

    Subclasses implement :meth:`load_model` and :meth:`synthesize`; a model that can
    produce audio incrementally also overrides :meth:`synthesize_stream`.
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

    def synthesize_stream(self, text: str) -> Iterator[tuple[Any, int]]:
        """Yield ``(float samples, sample rate)`` chunks for ``text`` as they are ready.

        Runs in a worker thread. The default yields :meth:`synthesize` as one chunk.
        """
        yield self.synthesize(text)

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
        chunks = iterate_in_thread(self.synthesize_stream(text))
        try:
            await self.ensure_loaded()
            await self.start_tts_usage_metrics(text)
            first = await anext(chunks, None)
            if first is None:
                logger.warning(f"{self}: no audio for [{text}]")
                return
            sample_rate = int(first[1])
            piece = max(2, int(sample_rate * _PUSH_SECS) * 2)
            sent = 0

            async def pieces() -> AsyncIterator[bytes]:
                nonlocal sent
                item: tuple[Any, int] | None = first
                while item is not None:
                    samples, rate = item
                    if int(rate) != sample_rate:
                        raise ValueError(f"sample rate changed mid-utterance: {rate}")
                    pcm = float_to_pcm16(samples)
                    sent += len(pcm)
                    for chunk in chunk_bytes(pcm, piece):
                        yield chunk
                    item = await anext(chunks, None)

            async for frame in self._stream_audio_frames_from_iterator(
                pieces(), in_sample_rate=sample_rate, context_id=context_id
            ):
                await self.stop_ttfb_metrics()
                yield frame
            if not sent:
                logger.warning(f"{self}: no audio for [{text}]")
        except Exception as e:
            logger.exception(f"{self}: synthesis failed for [{text}]")
            yield ErrorFrame(error=f"{type(self).__name__} synthesis failed: {e}")
        finally:
            await chunks.aclose()
            await self.stop_ttfb_metrics()
