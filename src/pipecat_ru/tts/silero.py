"""Silero v5 Russian TTS for Pipecat: a fast CPU fallback.

`Silero models <https://github.com/snakers4/silero-models>`_ synthesize Russian
quickly on a CPU, which makes them a good fallback when the GPU is busy or down.

.. warning::
   The Silero TTS **weights are licensed CC BY-NC 4.0 (non-commercial)**. This code
   is Apache-2.0, but a product that ships or serves Silero voices commercially needs
   a licence from the Silero authors.

Install::

    pip install "pipecat-ru[silero]"

Speakers of the ``v5_ru`` model: ``aidar``, ``baya``, ``kseniya``, ``xenia``,
``eugene``. Sample rates: 8000, 24000, 48000. Silero understands ``+`` stress marks
(``з+амок``), so it pairs well with a normalizer that has a
:class:`~pipecat_ru.text.StressMarker` in ``"plus"`` format.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language

from pipecat_ru.tts._base import RussianLocalTTSService

SILERO_SAMPLE_RATES = (8000, 24000, 48000)
SILERO_SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")


def load_silero_tts(model_id: str = "v5_ru", *, device: str = "cpu") -> Any:
    """Load a Silero TTS model via the ``silero`` package, or torch.hub as a fallback."""
    try:
        import torch
    except ModuleNotFoundError as e:  # pragma: no cover - depends on extras
        raise ImportError('Silero TTS needs torch: pip install "pipecat-ru[silero]"') from e
    try:
        from silero import silero_tts

        model, _ = silero_tts(language="ru", speaker=model_id)
    except ModuleNotFoundError:  # pragma: no cover - depends on extras
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=model_id,
            trust_repo=True,
        )
    model.to(torch.device(device))
    return model


class SileroTTSService(RussianLocalTTSService):
    """Pipecat TTS service running Silero v5 Russian (non-commercial weights).

    Args:
        speaker: One of :data:`SILERO_SPEAKERS`.
        model_id: Silero model id (``"v5_ru"``; newer ``v5_x_ru`` ids also work).
        silero_sample_rate: Synthesis rate, one of 8000, 24000, 48000. Pipecat
            resamples to the pipeline's output rate.
        device: ``"cpu"`` (default) or ``"cuda"``.
        num_threads: Torch CPU threads (None leaves torch's default).
        put_accent: Let Silero place stress automatically.
        put_yo: Let Silero restore "ё".
        model: An already loaded Silero model (or a test double).
        **kwargs: Passed to :class:`~pipecat_ru.tts._base.RussianLocalTTSService`
            (``normalizer``, ``splitter``) and Pipecat's ``TTSService``.
    """

    def __init__(
        self,
        *,
        speaker: str = "xenia",
        model_id: str = "v5_ru",
        silero_sample_rate: int = 48000,
        device: str = "cpu",
        num_threads: int | None = None,
        put_accent: bool = True,
        put_yo: bool = True,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        """Create the service; see the class docstring for the arguments."""
        if silero_sample_rate not in SILERO_SAMPLE_RATES:
            raise ValueError(f"silero_sample_rate must be one of {SILERO_SAMPLE_RATES}")
        kwargs.setdefault(
            "settings", TTSSettings(model=model_id, voice=speaker, language=Language.RU)
        )
        super().__init__(**kwargs)
        self._speaker = speaker
        self._model_id = model_id
        self._silero_sample_rate = silero_sample_rate
        self._silero_device = device
        self._num_threads = num_threads
        self._put_accent = put_accent
        self._put_yo = put_yo
        self._silero_model = model
        self._extra_args_supported = True

    def language_to_service_language(self, language: Language) -> str | None:
        """This service speaks Russian only."""
        return "ru"

    def load_model(self) -> None:
        """Load the model (once)."""
        if self._num_threads:
            import torch

            torch.set_num_threads(self._num_threads)
        if self._silero_model is None:
            logger.info(f"{self}: loading Silero {self._model_id} on {self._silero_device}...")
            self._silero_model = load_silero_tts(self._model_id, device=self._silero_device)

    def synthesize(self, text: str) -> tuple[Any, int]:
        """Synthesize one chunk."""
        args: dict[str, Any] = {
            "text": text,
            "speaker": self._speaker,
            "sample_rate": self._silero_sample_rate,
        }
        if self._extra_args_supported:
            try:
                audio = self._silero_model.apply_tts(
                    **args, put_accent=self._put_accent, put_yo=self._put_yo
                )
                return audio, self._silero_sample_rate
            except TypeError:
                # some Silero builds do not take put_accent/put_yo; stop passing them
                self._extra_args_supported = False
        return self._silero_model.apply_tts(**args), self._silero_sample_rate
